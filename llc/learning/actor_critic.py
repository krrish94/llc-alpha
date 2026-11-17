"""Short-horizon actor-critic through the plant (deep dive, item 1 of the gradient plan).

Structure follows SHAC / SuperMimic: the policy is optimized by backpropagation through time over
chunks of ``H`` steps of the MuJoCo-gradient rollout, each chunk's return bootstrapped by a critic
at its last state, and the critic is fit to TD(lambda) targets computed backward over the whole
window. Rewards are bounded exponential tracking kernels (whose gradient vanishes for targets that
cannot be tracked) plus the torque, torque-rate and contact-force penalties of ``LossConfig``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from llc.learning.policy import HistoryBuffer, Policy, OutputFilter, preview_indices
from llc.learning.rollout_mj import _clip_hook
from llc.sim.panda import NUM_JOINTS


@dataclass
class RewardConfig:
    pos_sigma: float = 0.05  # rad, half-width of the position-tracking kernel (per-joint RMS)
    vel_sigma: float = 0.5  # rad/s
    w_pos: float = 1.0
    w_vel: float = 0.3
    w_torque: float = 0.02
    w_rate: float = 0.03
    rate_scale: float = 500.0  # Nm/s
    w_force: float = 0.5
    force_soft: float = 30.0  # N
    force_scale: float = 20.0  # N
    w_impulse: float = 0.02
    impulse_scale: float = 2000.0  # N/s
    w_lin: float = 0.1  # weak linear restoring term so that large errors keep a bounded gradient
    lin_scale: float = 0.05  # rad
    gamma: float = 0.995  # per 1 ms step (effective horizon 200 ms; the chunk's own rewards and the bootstrap are comparable)
    lam: float = 0.95
    reward_scale: float = 1e-2  # keeps discounted returns O(1)
    future_offsets_ms: tuple[int, ...] = (50, 100, 200, 500)  # reference preview given to the critic (privileged)
    # Contact-maintenance terms (ADR 0007; privileged, simulation only), active while the reference is
    # inside a surface: a kernel on the normal force around the force the commanded depth implies
    # (press_min + press_stiffness * depth, capped at force_cap), the tangential tool-tracking kernel,
    # and the joint kernel switched off (it would ask for the infeasible depth). w_press = 0 disables.
    w_press: float = 0.0
    press_sigma: float = 5.0  # N
    press_stiffness: float = 500.0  # N/m: 1 cm of commanded depth asks for 5 N above press_min (ADR 0008: the oracle's spring; 1000 in ADR 0007)
    press_min: float = 5.0  # N
    force_cap: float = 15.0  # N
    w_feas: float = 1.0
    feas_sigma: float = 0.01  # m
    # Force terms on a first-order low-pass of the contact force (seconds; 0 = instantaneous, D3): a
    # blind controller cannot avoid the impact transient at the reference speed, and a penalty on it
    # 100 times the tracking reward teaches softening everywhere; the sustained force is what the cap
    # limits (as the hardware reflex and the f_peak_20ms metric, ADR 0007).
    force_tau: float = 0.0
    # "full": the actor gradient flows through the contact force (finite differences through the
    # contact); "detach": the force terms are rewards without a gradient path, the actor learns about
    # forces only through the critic's bootstrap (as a policy-gradient method would; ADR 0007).
    force_grad: str = "full"


def step_reward(e, ed, tau, last_tau, touch, prev_touch, limits, cfg: RewardConfig, dt: float) -> torch.Tensor:
    r = cfg.w_pos * torch.exp(-(e * e).mean(-1) / cfg.pos_sigma**2) + cfg.w_vel * torch.exp(-(ed * ed).mean(-1) / cfg.vel_sigma**2)
    r = r - cfg.w_torque * (tau / limits).square().sum(-1) - cfg.w_rate * ((tau - last_tau) / (dt * cfg.rate_scale)).square().sum(-1)
    r = r - cfg.w_force * (torch.relu(touch - cfg.force_soft) / cfg.force_scale).square() - cfg.w_impulse * ((touch - prev_touch) / (dt * cfg.impulse_scale)).square()
    r = r - cfg.w_lin * (torch.sqrt(e * e + 1e-4).mean(-1) / cfg.lin_scale)
    return r * cfg.reward_scale


def contact_reward(cfg: RewardConfig, e_joint, tool_pos, tool_ref, normal, depth_ref, has_surface, touch, tan_ref=None) -> torch.Tensor:
    """Contact-maintenance terms (ADR 0007), (B,). Active while the reference is inside a surface:
    tangential tool kernel + force kernel around the depth-implied, capped target force - the joint
    kernel of ``step_reward`` (so that the infeasible depth is not asked for). A lost contact (touch 0)
    scores ~0 on the force kernel, which is what charges contact loss. ``tan_ref`` (B,3) replaces the
    tangential target (the first-contact anchor of blocked windows, ``BlockedAnchor``)."""
    inside = (has_surface & (depth_ref > 0.0)).float()
    e_tool = (tool_ref if tan_ref is None else tan_ref) - tool_pos
    e_tan = e_tool - normal * (normal * e_tool).sum(-1, keepdim=True)
    k_tan = torch.exp(-(e_tan * e_tan).sum(-1) / cfg.feas_sigma**2)
    f_target = torch.clamp(cfg.press_min + cfg.press_stiffness * depth_ref, max=cfg.force_cap)
    k_force = torch.exp(-((touch - f_target) / cfg.press_sigma).square())
    k_joint = torch.exp(-(e_joint * e_joint).mean(-1) / cfg.pos_sigma**2)
    return cfg.reward_scale * inside * (cfg.w_feas * k_tan + cfg.w_press * k_force - cfg.w_pos * k_joint)


class BlockedAnchor:
    """Stop at the surface and wait (ADR 0007 amendment, ADR 0008): in windows of the blocked family
    the tangential target of ``contact_reward`` is the tool position at the first contact while the
    reference is inside, instead of the reference's tangential projection (which asks for a slide)."""

    def __init__(self, labels, has_surface: torch.Tensor):
        self.blocked = torch.tensor([lab == "blocked" for lab in labels]) & has_surface.bool()
        self.anchor = torch.zeros(len(labels), 3)
        self.set = torch.zeros(len(labels), dtype=torch.bool)

    def tangential_target(self, tool_pos: torch.Tensor, tool_ref: torch.Tensor, depth_ref: torch.Tensor, touch: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        new = self.blocked & ~self.set & (depth_ref > 0.0) & (touch > threshold)
        self.anchor[new] = tool_pos[new]
        self.set |= new
        return torch.where(self.set[:, None], self.anchor, tool_ref)


def reference_accel(qd_ref: torch.Tensor, dt: float, clip: float = 50.0) -> torch.Tensor:
    """Reference acceleration of a window (B,T,7) by central differences of the reference velocity, clipped
    (held targets keep a zero velocity, so their jumps do not produce spikes)."""
    qdd = torch.zeros_like(qd_ref)
    qdd[:, 1:-1] = (qd_ref[:, 2:] - qd_ref[:, :-2]) / (2.0 * dt)
    return qdd.clamp(-clip, clip)


def base_term(policy: Policy, gravity_model, q_obs, qd_ref_k, qdd_ref_k, gravity):
    """The PD base's model term at a physics step: the inverse-dynamics feedforward when the policy carries it, else gravity."""
    if getattr(policy.cfg, "feedforward", False):
        return gravity_model.inverse_dynamics(q_obs, qd_ref_k, qdd_ref_k)
    return gravity


def priv_tensors(priv: dict | None) -> dict | None:
    return None if priv is None else {k: torch.as_tensor(v) for k, v in priv.items()}


class Critic(nn.Module):
    """V(state, goal, privileged inputs): current position, velocity, tracking error and reference
    velocity, the reference preview, the touch force and the payload mass. Deliberately no action
    history: a value function that sees the past torques gives the actor a direct, noisy gradient
    channel through its own inputs (observed as early collapse)."""

    def __init__(self, obs_dim: int, n_future: int, hidden: int = 256):
        super().__init__()
        self.priv_dim = 2 + n_future * NUM_JOINTS
        self.state_dim = 4 * NUM_JOINTS
        self.net = nn.Sequential(nn.Linear(self.state_dim + self.priv_dim, hidden), nn.LayerNorm(hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def critic_input(obs, touch, payload, q, q_ref, k, offsets_steps, T):
    """State part of the policy observation (first 28 numbers: q, qd, error, reference velocity)
    plus the privileged inputs at step k."""
    fut = [(q_ref[:, min(T - 1, k + o)] - q) / 0.2 for o in offsets_steps]
    return torch.cat([obs[:, :4 * NUM_JOINTS], (touch / 50.0)[:, None], (payload / 3.0)[:, None]] + fut, -1)


def rollout_ac(policy: Policy, critic: Critic, batch, gravity_model, q_ref, qd_ref, q0, qd0, payload, f_ext=None, rcfg: RewardConfig = RewardConfig(),
               horizon: int = 100, train: bool = False, obs_noise_q: float = 1e-4, dt: float = 1e-3, state_grad_clip: float = 1e3,
               median_clip: float = 0.0, bootstrap_scale: float = 1.0, priv: dict | None = None, labels=None) -> dict:
    """Chunked rollout with actor updates per chunk (if ``train``), returning the data for the critic fit.
    ``payload`` (B,) in kg is a privileged critic input. ``median_clip`` > 0 replaces the fixed state
    gradient clip by a per-sample clip at ``median_clip`` times the batch median norm. ``priv`` (from
    ``sample_batch_v2``) enables the contact-maintenance terms when ``rcfg.w_press`` > 0; the tool
    position then comes from the differentiable rigid-body model."""
    B, T, _ = q_ref.shape
    limits = policy.limits
    offsets = [int(round(ms / (dt * 1000))) for ms in rcfg.future_offsets_ms]
    batch.reset(np.asarray(q0, dtype=np.float64), np.asarray(qd0, dtype=np.float64))
    q, qd = (x.float() for x in batch.state())
    buffer = HistoryBuffer(policy.cfg, q, qd, gravity_model.inverse_dynamics if policy.cfg.ext_torque else None, dt)
    out_filter = OutputFilter(policy.cfg.output_tau, dt)
    pv = priv_tensors(priv) if rcfg.w_press > 0 else None
    anchor = BlockedAnchor(labels, pv["has_surface"]) if pv is not None and labels is not None else None
    alpha_f = min(1.0, dt / rcfg.force_tau) if rcfg.force_tau > 0 else 1.0  # reward-side force filter
    touch_f = torch.zeros(B)
    last_tau = torch.zeros(B, NUM_JOINTS)
    touch = torch.zeros(B)
    payload_t = torch.as_tensor(payload, dtype=torch.float32)
    qdd_ref = reference_accel(qd_ref, dt) if getattr(policy.cfg, "feedforward", False) else None
    for p in critic.parameters():
        p.requires_grad_(False)
    logs = {k: [] for k in ("q", "qd", "tau", "touch", "tool_pos", "reward")}
    crit_in, rewards = [], []
    stats = {"reward": 0.0, "kernel_pos": 0.0, "force_pen": 0.0, "actor_loss": 0.0}
    hook = (lambda g: _median_clip_hook(g, median_clip)) if median_clip > 0 else _clip_hook(state_grad_clip)
    for k0 in range(0, T, horizon):
        k1 = min(T, k0 + horizon)
        ret = torch.zeros(B)
        disc = 1.0
        for k in range(k0, k1):
            q_obs = q + obs_noise_q * torch.randn_like(q) if obs_noise_q > 0 else q
            qd_obs = qd + 10.0 * obs_noise_q * torch.randn_like(qd) if obs_noise_q > 0 else qd
            buffer.push(q_obs, qd_obs, last_tau, q_ref[:, k])
            obs = buffer.observation(q_obs, qd_obs, q_ref[:, k], qd_ref[:, k], limits, q_ref[:, preview_indices(policy.cfg, k, T)] if policy.cfg.preview else None)
            crit_in.append(critic_input(obs, touch, payload_t, q, q_ref, k, offsets, T).detach())
            held = out_filter(policy.decode(policy(obs)))
            gravity = gravity_model.bias(q_obs, torch.zeros_like(q_obs))
            base = base_term(policy, gravity_model, q_obs, qd_ref[:, k], None if qdd_ref is None else qdd_ref[:, k], gravity)
            tau = policy.fast_torque(held, q_obs, qd_obs, q_ref[:, k], qd_ref[:, k], base)
            tau = torch.maximum(torch.minimum(tau, limits), -limits)
            e, ed = q_ref[:, k] - q, qd_ref[:, k] - qd
            q_n, qd_n, touch_n, tool_pos = batch.step(q, qd, tau, None if f_ext is None else f_ext[:, k])
            q_n, qd_n, touch_n = q_n.float(), qd_n.float(), touch_n.float()
            if train and q_n.requires_grad:
                q_n.register_hook(hook); qd_n.register_hook(hook)
            kn = min(T - 1, k + 1)
            e_n, ed_n = q_ref[:, kn] - q_n, qd_ref[:, kn] - qd_n
            touch_fn = touch_f + alpha_f * (touch_n - touch_f) if alpha_f < 1.0 else touch_n
            touch_r = touch_fn.detach() if rcfg.force_grad == "detach" else touch_fn
            r = step_reward(e_n, ed_n, tau, last_tau, touch_r, touch_f.detach() if rcfg.force_grad == "detach" else touch_f, limits, rcfg, dt)
            if pv is not None:
                tp = gravity_model.tool_pos(q_n)
                tan = anchor.tangential_target(torch.as_tensor(tool_pos, dtype=torch.float32), pv["tool_ref"][:, kn], pv["depth_ref"][:, kn], touch_n) if anchor is not None else None
                r = r + contact_reward(rcfg, e_n, tp, pv["tool_ref"][:, kn], pv["normal"], pv["depth_ref"][:, kn], pv["has_surface"], touch_r, tan)
            touch_f = touch_fn
            ret = ret + disc * r
            disc *= rcfg.gamma
            rewards.append(r.detach())
            stats["kernel_pos"] += torch.exp(-(e_n * e_n).mean(-1) / rcfg.pos_sigma**2).mean().item() / T
            stats["force_pen"] += (torch.relu(touch_n - rcfg.force_soft) / rcfg.force_scale).square().mean().item() / T
            last_tau, touch, q, qd = tau, touch_n, q_n, qd_n
            for name, val in (("q", q), ("qd", qd), ("tau", tau), ("touch", touch), ("tool_pos", torch.as_tensor(tool_pos))):
                logs[name].append(val.detach())
        # bootstrap the chunk return with the critic at the chunk's last state (gradient flows into the state)
        q_obs = q + obs_noise_q * torch.randn_like(q) if obs_noise_q > 0 else q
        qd_obs = qd + 10.0 * obs_noise_q * torch.randn_like(qd) if obs_noise_q > 0 else qd
        obs_end = buffer.observation(q_obs, qd_obs, q_ref[:, min(T - 1, k1)], qd_ref[:, min(T - 1, k1)], limits, q_ref[:, preview_indices(policy.cfg, min(T - 1, k1), T)] if policy.cfg.preview else None)
        v_end = critic(critic_input(obs_end, touch, payload_t, q, q_ref, min(T - 1, k1), offsets, T))
        actor_loss = -(ret + bootstrap_scale * disc * v_end).mean()
        if train:
            actor_loss.backward()
        stats["actor_loss"] += actor_loss.item() * (k1 - k0) / T
        q, qd, last_tau, touch, touch_f = q.detach(), qd.detach(), last_tau.detach(), touch.detach(), touch_f.detach()
        buffer.detach(); out_filter.detach()
    for p in critic.parameters():
        p.requires_grad_(True)
    # final critic input for the terminal bootstrap of the TD(lambda) targets
    obs_T = buffer.observation(q, qd, q_ref[:, T - 1], qd_ref[:, T - 1], limits, q_ref[:, preview_indices(policy.cfg, T - 1, T)] if policy.cfg.preview else None)
    crit_in.append(critic_input(obs_T, touch, payload_t, q, q_ref, T - 1, offsets, T).detach())
    out = {name: torch.stack(vals, 1) for name, vals in logs.items() if vals}
    out["critic_inputs"] = torch.stack(crit_in, 1)  # (B, T+1, D)
    out["rewards"] = torch.stack(rewards, 1)  # (B, T)
    stats["reward"] = float(out["rewards"].sum(1).mean())
    out["stats"] = stats
    return out


def _median_clip_hook(g, kappa: float):
    if g is None:
        return None
    n = g.norm(dim=-1, keepdim=True)
    med = n.median()
    return g * torch.clamp(kappa * med / (n + 1e-12), max=1.0)


def td_lambda_targets(critic: Critic, crit_in: torch.Tensor, rewards: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """TD(lambda) returns computed backward over the window; (B, T)."""
    with torch.no_grad():
        B, T1, D = crit_in.shape
        v = critic(crit_in.reshape(-1, D)).reshape(B, T1)
    T = rewards.shape[1]
    R = torch.zeros(B, T)
    nxt = v[:, T]
    for t in range(T - 1, -1, -1):
        nxt = rewards[:, t] + gamma * ((1 - lam) * v[:, t + 1] + lam * nxt)
        R[:, t] = nxt
    return R


def fit_critic(critic: Critic, opt: torch.optim.Optimizer, crit_in: torch.Tensor, targets: torch.Tensor, epochs: int = 5, minibatches: int = 4) -> float:
    x = crit_in[:, :-1].reshape(-1, crit_in.shape[-1])
    y = targets.reshape(-1)
    n = len(x)
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(minibatches):
            idx = perm[i * n // minibatches:(i + 1) * n // minibatches]
            loss = (critic(x[idx]) - y[idx]).square().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
            last = loss.item()
    return last
