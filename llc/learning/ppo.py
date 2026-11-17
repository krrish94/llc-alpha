"""PPO in the MuJoCo plant for the residual controller (deep dive D4).

The environment is the batch of plant simulators of the deep dive (``MjBatch`` without gradients):
protocol v2 training windows with per-window robot instances, payloads and pulses. One decision of
the policy is held for ``hold_steps`` physics steps (1 ms each) while the proprioceptive history is
still sampled at every physics step, so a policy trained at 1 kHz keeps the meaning of its inputs
when it is fine-tuned at a lower decision rate. The reward is the per-step reward of the
actor-critic objective (bounded tracking kernels minus torque, torque-rate and contact-force
penalties) summed over the held steps, so a critic trained there is a valid warm start.

A conservative force reflex models the real robot's collision reaction: if the measured normal
contact force exceeds ``reflex_force`` the window ends for that robot with a penalty, the residual
is switched off and the remaining steps of the window are excluded from the update.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from llc.learning.actor_critic import BlockedAnchor, Critic, RewardConfig, critic_input, step_reward, contact_reward, priv_tensors
from llc.learning.data import sample_batch_v2
from llc.learning.policy import HistoryBuffer, Policy, OutputFilter, preview_indices
from llc.sim.instances import sample_instance
from llc.sim.nominal import NominalModel

LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class PPOConfig:
    hold_steps: int = 10  # decision every 10 ms
    gamma_step: float = 0.995  # discount per physics step (as the actor-critic objective)
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatch: int = 2048
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    init_std: float = 0.1  # in the pre-tanh action space (the residual is 0.5 tau_max tanh(a))
    entropy: float = 0.0
    max_grad_norm: float = 0.5
    reflex_force: float = 40.0  # N
    reflex_tau: float = 0.0  # s; > 0: the reflex acts on a first-order low-pass of the force (sustained force), 0: instantaneous
    reflex_penalty: float = 5.0  # reward units; a 2 s window of perfect tracking is about 26
    reflex_terminate: bool = True  # False: the penalty is charged at each crossing but the window continues (post-impact behaviour is trained)
    obs_noise_q: float = 1e-4
    # Privileged feasible-direction tracking (simulation only): in windows with a surface, while the tool is
    # in contact, the joint-space position kernel is replaced by a kernel on the tool-space error with its
    # component along the surface normal removed ("track what is feasible, yield along the normal").
    feasible_kernel: bool = False
    feas_sigma: float = 0.01  # m
    w_feas: float = 1.0
    contact_threshold: float = 1.0  # N: force above which the normal component of the tool error stops being rewarded
    reg_warmstart: float = 0.0  # weight of an L2 penalty between the actor mean and the frozen warm-start mean (limits drift in states the windows rarely visit)
    start_mode: str = "contact"  # window starts of the surface families: "contact" (before the onset) or "mixed" (half before the release; ADR 0007)
    noise_rho: float = 0.0  # temporal correlation of the exploration noise between decisions (0 = white; ADR 0007)

    @property
    def gamma(self) -> float:
        return self.gamma_step**self.hold_steps


class GaussianActor(nn.Module):
    """Diagonal Gaussian over the pre-tanh residual output of a ``Policy``; the mean is the policy."""

    def __init__(self, policy: Policy, init_std: float, reference: Policy | None = None):
        super().__init__()
        self.policy = policy
        self.log_std = nn.Parameter(torch.full((policy.cfg.out_dim,), float(np.log(init_std))))
        self.reference = reference  # frozen warm-start policy for the drift penalty (not a submodule: no gradients, not saved)
        if reference is not None:
            for prm in reference.parameters():
                prm.requires_grad_(False)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(obs), self.log_std.exp().expand(obs.shape[0], -1)

    def log_prob(self, mean: torch.Tensor, std: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return (-0.5 * ((act - mean) / std).square() - torch.log(std) - 0.5 * LOG_2PI).sum(-1)

    def entropy(self) -> torch.Tensor:
        return (0.5 + 0.5 * LOG_2PI + self.log_std).sum()


class PPOEnv:
    """B plant windows advancing one decision per ``step``; observations as the learned controller sees them."""

    def __init__(self, batch, policy: Policy, gravity_model, pool: dict, weights: dict, horizon_steps: int, rng: np.random.Generator,
                 rcfg: RewardConfig, pcfg: PPOConfig, dt: float = 1e-3):
        self.batch, self.policy, self.gravity_model, self.pool, self.weights = batch, policy, gravity_model, pool, weights
        self.T, self.rng, self.rcfg, self.pcfg, self.dt = horizon_steps, rng, rcfg, pcfg, dt
        self.B = batch.B
        self.limits = policy.limits
        self.offsets = [int(round(ms / (dt * 1000))) for ms in rcfg.future_offsets_ms]
        self.decisions = horizon_steps // pcfg.hold_steps
        self.nominal = NominalModel() if pcfg.feasible_kernel else None

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        q_ref, qd_ref, q0, qd0, labels, f_ext, priv = sample_batch_v2(self.pool, self.weights, self.B, self.T, self.dt, self.rng, start_mode=self.pcfg.start_mode)
        self.priv = priv_tensors(priv) if self.rcfg.w_press > 0 else None
        self.anchor = BlockedAnchor(labels, self.priv["has_surface"]) if self.priv is not None else None  # stop and wait: blocked windows hold the first-contact point
        self.payloads = np.array([float(self.rng.uniform(0, 3)) for _ in range(self.B)], np.float32)
        self.batch.perturb([sample_instance(self.rng, payload_mass=float(self.payloads[i]), payload_offset=(self.rng.uniform(-0.03, 0.03), self.rng.uniform(-0.03, 0.03), 0.10))
                            for i in range(self.B)])
        self.batch.reset(np.asarray(q0, dtype=np.float64), np.asarray(qd0, dtype=np.float64))
        self.q_ref, self.qd_ref, self.f_ext, self.labels = torch.as_tensor(q_ref), torch.as_tensor(qd_ref), f_ext, labels
        if self.pcfg.feasible_kernel:  # tool-space reference at decision resolution and surface normals (privileged)
            h = self.pcfg.hold_steps
            self.tool_ref = torch.as_tensor(np.stack([[self.nominal.tool_pos(q_ref[b, k]) for k in range(0, self.T, h)] for b in range(self.B)]), dtype=torch.float32)
            normals = np.zeros((self.B, 3), np.float32)
            self.has_normal = torch.zeros(self.B, dtype=torch.bool)
            for b, lab in enumerate(labels):
                n = self.pool[lab][0]["meta"].get("normal") if lab in ("contact2", "blocked") else None
                if n is not None:
                    normals[b] = np.asarray(n, np.float32); self.has_normal[b] = True
            self.normals = torch.as_tensor(normals)
        self.payload_t = torch.as_tensor(self.payloads)
        self.q, self.qd = (x.float() for x in self.batch.state())
        self.buffer = HistoryBuffer(self.policy.cfg, self.q, self.qd, self.gravity_model.inverse_dynamics if self.policy.cfg.ext_torque else None, self.dt)
        self.out_filter = OutputFilter(self.policy.cfg.output_tau, self.dt)
        self.last_tau = torch.zeros(self.B, self.limits.shape[0])
        self.touch = torch.zeros(self.B)
        self.touch_filt = torch.zeros(self.B)
        self.touch_rf = torch.zeros(self.B)  # reward-side filtered force (rcfg.force_tau)
        self.k = 0
        self.done = torch.zeros(self.B, dtype=torch.bool)
        self.reflexed = torch.zeros(self.B, dtype=torch.bool)  # any crossing in the window (non-terminating mode)
        self.above = torch.zeros(self.B, dtype=torch.bool)
        self.max_touch = torch.zeros(self.B)
        return self._observe()

    def _noisy(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.pcfg.obs_noise_q
        if n <= 0:
            return self.q, self.qd
        return self.q + n * torch.randn_like(self.q), self.qd + 10.0 * n * torch.randn_like(self.qd)

    def _observe(self) -> tuple[torch.Tensor, torch.Tensor]:
        kk = min(self.k, self.T - 1)
        q_obs, qd_obs = self._noisy()
        self.buffer.push(q_obs, qd_obs, self.last_tau, self.q_ref[:, kk])
        obs = self.buffer.observation(q_obs, qd_obs, self.q_ref[:, kk], self.qd_ref[:, kk], self.limits, self.q_ref[:, preview_indices(self.policy.cfg, kk, self.T)] if self.policy.cfg.preview else None)
        crit = critic_input(obs, self.touch, self.payload_t, self.q, self.q_ref, kk, self.offsets, self.T)
        self._q_obs, self._qd_obs = q_obs, qd_obs
        return obs, crit

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        """Apply a pre-tanh action for one decision. Returns (obs, critic input, reward, newly done, truncated)."""
        held = self.policy.decode(action)
        held = {k: v * (~self.done)[:, None].float() for k, v in held.items()}  # reflexed robots: PD base only
        reward = torch.zeros(self.B)
        newly_done = torch.zeros(self.B, dtype=torch.bool)
        for s in range(self.pcfg.hold_steps):
            if s == 0:
                q_obs, qd_obs = self._q_obs, self._qd_obs
            else:
                q_obs, qd_obs = self._noisy()
                self.buffer.push(q_obs, qd_obs, self.last_tau, self.q_ref[:, self.k])
            gravity = self.gravity_model.bias(q_obs, torch.zeros_like(q_obs))
            if getattr(self.policy.cfg, "feedforward", False):
                from llc.learning.actor_critic import reference_accel

                gravity = self.gravity_model.inverse_dynamics(q_obs, self.qd_ref[:, self.k], reference_accel(self.qd_ref, self.dt)[:, self.k])
            tau = self.policy.fast_torque(self.out_filter(held), q_obs, qd_obs, self.q_ref[:, self.k], self.qd_ref[:, self.k], gravity)
            tau = torch.maximum(torch.minimum(tau, self.limits), -self.limits)
            q_n, qd_n, touch_n, tool_pos = self.batch.step(self.q, self.qd, tau, self.f_ext[:, self.k])
            q_n, qd_n, touch_n = q_n.float(), qd_n.float(), touch_n.float()
            kn = min(self.T - 1, self.k + 1)
            e_n = self.q_ref[:, kn] - q_n
            if self.rcfg.force_tau > 0:
                touch_rf = self.touch_rf + min(1.0, self.dt / self.rcfg.force_tau) * (touch_n - self.touch_rf)
            else:
                touch_rf = touch_n
            r = step_reward(e_n, self.qd_ref[:, kn] - qd_n, tau, self.last_tau, touch_rf, self.touch_rf, self.limits, self.rcfg, self.dt)
            if self.priv is not None:
                tp = torch.as_tensor(tool_pos, dtype=torch.float32)
                tan = self.anchor.tangential_target(tp, self.priv["tool_ref"][:, kn], self.priv["depth_ref"][:, kn], touch_n)
                r = r + contact_reward(self.rcfg, e_n, tp, self.priv["tool_ref"][:, kn], self.priv["normal"],
                                       self.priv["depth_ref"][:, kn], self.priv["has_surface"], touch_rf, tan)
            self.touch_rf = touch_rf
            if self.pcfg.feasible_kernel:
                c = (self.has_normal & (touch_n > self.pcfg.contact_threshold)).float()[:, None]
                e_tool = self.tool_ref[:, min(kn // self.pcfg.hold_steps, self.tool_ref.shape[1] - 1)] - torch.as_tensor(tool_pos, dtype=torch.float32)
                e_feas = e_tool - c * self.normals * (self.normals * e_tool).sum(-1, keepdim=True)
                k_joint = torch.exp(-(e_n * e_n).mean(-1) / self.rcfg.pos_sigma**2)
                k_tool = torch.exp(-(e_feas * e_feas).sum(-1) / self.pcfg.feas_sigma**2)
                r = r + self.rcfg.reward_scale * c[:, 0] * (self.pcfg.w_feas * k_tool - self.rcfg.w_pos * k_joint)
            active = ~self.done
            reward += r * active.float()
            if self.pcfg.reflex_tau > 0:
                a = self.dt / self.pcfg.reflex_tau
                self.touch_filt = self.touch_filt + a * (touch_n - self.touch_filt)
                over = self.touch_filt > self.pcfg.reflex_force
            else:
                over = touch_n > self.pcfg.reflex_force
            reflex = active & over & ~self.above  # rising edge
            reward[reflex] -= self.pcfg.reflex_penalty
            if self.pcfg.reflex_terminate:
                self.done |= reflex
                newly_done |= reflex
            else:  # count the crossing (rising edge of the filtered force) but keep training the window
                self.reflexed |= reflex
                self.above = (self.touch_filt if self.pcfg.reflex_tau > 0 else touch_n) > self.pcfg.reflex_force
            self.max_touch = torch.maximum(self.max_touch, touch_n)
            self.last_tau, self.touch, self.q, self.qd = tau, touch_n, q_n, qd_n
            self.k += 1
        truncated = self.k >= self.T
        obs, crit = self._observe()
        return obs, crit, reward, newly_done, truncated


def collect(env: PPOEnv, actor: GaussianActor, critic: Critic) -> dict:
    """One window of every robot; returns tensors shaped (decisions, B, ...) plus masks and the bootstrap value."""
    obs, crit = env.reset()
    buf = {k: [] for k in ("obs", "crit", "act", "logp", "rew", "val", "valid", "done")}
    rho = env.pcfg.noise_rho
    eps = None
    with torch.no_grad():
        for _ in range(env.decisions):
            mean, std = actor(obs)
            white = torch.randn_like(mean)
            eps = white if (eps is None or rho <= 0) else rho * eps + float(np.sqrt(1.0 - rho * rho)) * white  # AR(1) exploration, unit variance
            act = mean + std * eps
            logp = actor.log_prob(mean, std, act)
            v = critic(crit)
            valid = ~env.done
            obs2, crit2, rew, newly_done, truncated = env.step(act)
            for k, x in (("obs", obs), ("crit", crit), ("act", act), ("logp", logp), ("rew", rew), ("val", v), ("valid", valid), ("done", newly_done)):
                buf[k].append(x)
            obs, crit = obs2, crit2
        v_last = critic(crit) * (~env.done).float()
    out = {k: torch.stack(v) for k, v in buf.items()}
    out["v_last"] = v_last
    out["labels"] = env.labels
    out["reflex"] = (env.done | env.reflexed).clone()
    out["max_touch"] = env.max_touch.clone()
    return out


def gae(roll: dict, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    rew, val, done = roll["rew"], roll["val"], roll["done"].float()
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(rew[0])
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        next_v = val[t + 1] if t + 1 < T else roll["v_last"]
        delta = rew[t] + gamma * next_v * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def ppo_update(actor: GaussianActor, critic: Critic, opt_actor, opt_critic, roll: dict, pcfg: PPOConfig) -> dict:
    adv, ret = gae(roll, pcfg.gamma, pcfg.lam)
    m = roll["valid"].flatten()
    obs, crit, act, logp_old = (roll[k].flatten(0, 1)[m] for k in ("obs", "crit", "act", "logp"))
    adv, ret = adv.flatten()[m], ret.flatten()[m]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(obs)
    stats = {"actor_loss": 0.0, "value_loss": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "n_minibatches": 0}
    for _ in range(pcfg.epochs):
        perm = torch.randperm(n)
        for i0 in range(0, n, pcfg.minibatch):
            idx = perm[i0:i0 + pcfg.minibatch]
            mean, std = actor(obs[idx])
            logp = actor.log_prob(mean, std, act[idx])
            ratio = torch.exp(logp - logp_old[idx])
            a = adv[idx]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - pcfg.clip, 1 + pcfg.clip) * a).mean()
            loss_actor = pg - pcfg.entropy * actor.entropy()
            if pcfg.reg_warmstart > 0 and actor.reference is not None:
                with torch.no_grad():
                    mean_ref = actor.reference(obs[idx])
                loss_actor = loss_actor + pcfg.reg_warmstart * (mean - mean_ref).square().sum(-1).mean()
            opt_actor.zero_grad(); loss_actor.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), pcfg.max_grad_norm)
            opt_actor.step()
            vl = 0.5 * (critic(crit[idx]) - ret[idx]).square().mean()
            opt_critic.zero_grad(); vl.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), pcfg.max_grad_norm)
            opt_critic.step()
            with torch.no_grad():
                stats["approx_kl"] += float((logp_old[idx] - logp).mean())
                stats["clip_frac"] += float(((ratio - 1).abs() > pcfg.clip).float().mean())
            stats["actor_loss"] += pg.item(); stats["value_loss"] += vl.item(); stats["n_minibatches"] += 1
    k = max(1, stats.pop("n_minibatches"))
    return {key: v / k for key, v in stats.items()} | {"n_samples": n, "valid_fraction": float(m.float().mean())}
