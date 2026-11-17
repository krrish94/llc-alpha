"""Milestone 5: train the same three controller parameterizations with PPO in the plant (MuJoCo),
with the curriculum of Milestone 2 and a reward equal to the negative training loss of ADR 0003.

    uv run python experiments/m5_ppo.py --kind learned_gains --seed 0 --out runs/m5/learned_gains-s0

Environment step = one network period (10 ms = 10 physics steps) with the same 5 ms latency and
history as the learned controller, so a trained PPO policy is exported as an ordinary policy
checkpoint and evaluated through the same code path as the model-based controllers.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from llc.controllers.pd import home_inertia_diag
from llc.eval.cases import REGIMES
from llc.eval.parallel import CaseSpec, evaluate
from llc.learning.data import reference_pool, sample_batch
from llc.learning.policy import Policy, PolicyConfig, encode_obs, save_policy
from llc.learning.rollout import LossConfig
from llc.sim.nominal import NominalModel
from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS, PandaSim

STAGES = [
    {"hold": 0.5, "smooth": 0.5},
    {"hold": 0.2, "smooth": 0.3, "zoh": 0.25, "step": 0.25},
    {"hold": 0.1, "smooth": 0.2, "zoh": 0.15, "step": 0.15, "contact": 0.2, "transition": 0.2},
]


class VecEnv:
    """N MuJoCo plants stepping one network period per call; observations as in the learned controller."""

    def __init__(self, n: int, policy: Policy, pool: dict, nominal: NominalModel, loss: LossConfig, horizon_s: float, rng: np.random.Generator):
        self.n, self.policy, self.pool, self.nominal, self.loss, self.rng = n, policy, pool, nominal, loss, rng
        self.dt = 1e-3
        self.period = int(round(1.0 / (policy.cfg.rate_hz * self.dt)))
        self.latency = int(round(policy.cfg.latency_s / self.dt))
        self.H = policy.cfg.history
        self.T = int(round(horizon_s / self.dt))
        self.sims = [PandaSim() for _ in range(n)]
        self.limits = TORQUE_LIMITS
        self.weights = STAGES[0]
        self.q_ref = np.zeros((n, self.T, 7), np.float32)
        self.qd_ref = np.zeros((n, self.T, 7), np.float32)
        self.k = np.zeros(n, int)
        self.hist_q = np.zeros((n, self.H, 7), np.float32)
        self.hist_qd = np.zeros((n, self.H, 7), np.float32)
        self.hist_tau = np.zeros((n, self.H, 7), np.float32)
        self.last_tau = np.zeros((n, 7), np.float32)
        self.held = [None] * n
        self.prev_f = np.zeros(n)
        for i in range(n):
            self.reset_one(i)

    def reset_one(self, i: int) -> None:
        q_ref, qd_ref, q0, qd0, _ = sample_batch(self.pool, self.weights, 1, self.T, self.dt, self.rng)
        self.q_ref[i], self.qd_ref[i] = q_ref[0], qd_ref[0]
        self.sims[i].reset(q0[0], qd0[0])
        self.k[i] = 0
        self.hist_q[i] = q0[0]; self.hist_qd[i] = qd0[0]; self.hist_tau[i] = 0.0
        self.last_tau[i] = 0.0
        self.held[i] = None
        self.prev_f[i] = 0.0

    def observe(self) -> torch.Tensor:
        q = np.stack([s.q for s in self.sims]); qd = np.stack([s.qd for s in self.sims])
        self.hist_q = np.concatenate([self.hist_q[:, 1:], q[:, None]], 1).astype(np.float32)
        self.hist_qd = np.concatenate([self.hist_qd[:, 1:], qd[:, None]], 1).astype(np.float32)
        self.hist_tau = np.concatenate([self.hist_tau[:, 1:], self.last_tau[:, None]], 1).astype(np.float32)
        idx = np.minimum(self.k, self.T - 1)
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return encode_obs(f(q), f(qd), f(self.q_ref[np.arange(self.n), idx]), f(self.qd_ref[np.arange(self.n), idx]), f(self.hist_q), f(self.hist_qd), f(self.hist_tau), self.policy.limits)

    def step(self, out: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Apply raw network outputs for one period. Returns rewards and done flags."""
        with torch.no_grad():
            new_held = {k: v.numpy() for k, v in self.policy.decode(out).items()}
        c = self.loss
        rewards, dones = np.zeros(self.n), np.zeros(self.n, bool)
        for i, sim in enumerate(self.sims):
            cost = 0.0
            for s in range(self.period):
                if s == self.latency:
                    self.held[i] = {k: v[i] for k, v in new_held.items()}
                held = self.held[i] if self.held[i] is not None else {k: v[0].numpy() for k, v in self.policy.neutral_output(1).items()}
                kk = min(self.k[i], self.T - 1)
                q, qd = sim.q, sim.qd
                qr, qdr = self.q_ref[i, kk], self.qd_ref[i, kk]
                if self.policy.cfg.kind == "learned_gains":
                    tau = held["kp"] * (qr - q) + held["kd"] * (qdr - qd) + self.nominal.gravity(q)
                elif self.policy.cfg.kind == "pd_residual":
                    tau = np.asarray(self.policy.cfg.kp_shared) * (qr - q) + np.asarray(self.policy.cfg.kd_shared) * (qdr - qd) + self.nominal.gravity(q) + held["residual"]
                else:
                    tau = held["torque"]
                tau = np.clip(tau, -self.limits, self.limits)
                sim.apply_tool_force(np.zeros(3))
                sim.step(tau)
                f_mag = sim.contact().f_normal
                e, ed = qr - sim.q, qdr - sim.qd
                cost += (np.sum((e / c.pos_scale) ** 2) + c.vel_weight * np.sum((ed / c.vel_scale) ** 2) + c.torque_weight * np.sum((tau / self.limits) ** 2)
                         + c.rate_weight * np.sum(((tau - self.last_tau[i]) / (self.dt * c.rate_scale)) ** 2)
                         + c.force_weight * (max(0.0, f_mag - c.force_soft) / c.force_scale) ** 2 + c.impulse_weight * ((f_mag - self.prev_f[i]) / (self.dt * c.impulse_scale)) ** 2)
                self.last_tau[i] = tau
                self.prev_f[i] = f_mag
                self.k[i] += 1
            rewards[i] = -cost / self.period
            diverged = not np.all(np.isfinite(sim.q)) or np.abs(sim.qd).max() > 50.0
            if diverged:
                rewards[i] -= 100.0
            dones[i] = diverged or self.k[i] >= self.T
            if dones[i]:
                self.reset_one(i)
        return rewards, dones


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def validate(ckpt: Path, n: int) -> dict:
    cases = [CaseSpec(r, "val", i) for r in REGIMES for i in range(n)]
    res = evaluate({"c": {"kind": "learned", "path": str(ckpt)}}, cases)["c"]
    per = {r: float(np.mean([m["composite"] for m in res if m["regime"] == r])) for r in REGIMES}
    return {"mean": float(np.mean(list(per.values()))), "per_regime": per, "diverged": int(sum(m["diverged"] for m in res))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--stage-updates", type=int, nargs=3, default=[60, 60, 120])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--init-std", type=float, default=0.3)
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--val-cases", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmstart", default="", help="policy checkpoint to initialize from (e.g. the PD imitation of neural torque)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    nominal = NominalModel()
    shared = json.load(open("runs/m1/gains_shared.json"))["spec"]
    cfg = PolicyConfig(kind=args.kind, inertia_diag=tuple(home_inertia_diag(nominal).tolist()), kp_shared=tuple(shared["kp"]), kd_shared=tuple(shared["kd"]))
    policy = Policy(cfg)
    if args.warmstart:
        from llc.learning.policy import load_policy

        policy.load_state_dict(load_policy(args.warmstart)[0].state_dict())
    log_std = nn.Parameter(torch.full((cfg.out_dim,), float(np.log(args.init_std))))
    value = ValueNet(cfg.obs_dim)
    opt = torch.optim.Adam([{"params": policy.parameters()}, {"params": [log_std]}, {"params": value.parameters()}], lr=args.lr)
    pool = reference_pool(32, 1e-3, "train")
    env = VecEnv(args.envs, policy, pool, nominal, LossConfig(), args.horizon, rng)
    steps_per_episode = env.T // env.period
    rollout_len = steps_per_episode
    obs_mean, obs_var, obs_count = torch.zeros(cfg.obs_dim), torch.ones(cfg.obs_dim), 1e-4
    ret_var, ret_run = 1.0, np.zeros(args.envs)
    git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    (out / "meta.json").write_text(json.dumps({"args": vars(args), "policy_config": cfg.__dict__, "stages": STAGES, "git": git, "env_steps_per_update": args.envs * rollout_len}, default=str))
    log_f = open(out / "log.jsonl", "a")
    best = {"val": float("inf"), "update": -1}
    t0, update, env_steps = time.time(), 0, 0
    total_updates = sum(args.stage_updates)
    obs = env.observe()
    for stage, n_updates in enumerate(args.stage_updates):
        env.weights = STAGES[stage]
        for _ in range(n_updates):
            buf = {k: [] for k in ("obs", "act", "logp", "rew", "done", "val")}
            for _ in range(rollout_len):
                with torch.no_grad():
                    nobs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-8)
                    mean = policy(nobs)
                    std = log_std.exp()
                    act = mean + std * torch.randn_like(mean)
                    logp = (-0.5 * ((act - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
                    v = value(nobs)
                rew, done = env.step(act)
                for k, val_ in (("obs", nobs), ("act", act), ("logp", logp), ("rew", torch.as_tensor(rew, dtype=torch.float32)), ("done", torch.as_tensor(done)), ("val", v)):
                    buf[k].append(val_)
                # running stats
                obs_count += len(obs); delta = obs.mean(0) - obs_mean; obs_mean += delta * len(obs) / obs_count; obs_var = 0.9995 * obs_var + 0.0005 * obs.var(0)
                ret_run = args.gamma * ret_run + rew; ret_var = 0.999 * ret_var + 0.001 * float(np.var(ret_run) + 1e-6)
                obs = env.observe()
                env_steps += args.envs
            with torch.no_grad():
                next_v = value((obs - obs_mean) / torch.sqrt(obs_var + 1e-8))
            rews = torch.stack(buf["rew"]) / np.sqrt(ret_var)
            vals = torch.stack(buf["val"] + [next_v])
            dones = torch.stack(buf["done"]).float()
            adv = torch.zeros_like(rews)
            last = torch.zeros(args.envs)
            for t in reversed(range(rollout_len)):
                nonterm = 1.0 - dones[t]
                delta = rews[t] + args.gamma * vals[t + 1] * nonterm - vals[t]
                last = delta + args.gamma * args.lam * nonterm * last
                adv[t] = last
            ret = adv + vals[:-1]
            b_obs, b_act, b_logp = torch.cat(buf["obs"]), torch.cat(buf["act"]), torch.cat(buf["logp"])
            b_adv, b_ret = adv.flatten(), ret.flatten()
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            n = len(b_obs)
            for _ in range(args.epochs):
                perm = torch.randperm(n)
                for i0 in range(0, n, args.minibatch):
                    idx = perm[i0:i0 + args.minibatch]
                    mean = policy(b_obs[idx]); std = log_std.exp()
                    logp = (-0.5 * ((b_act[idx] - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
                    ratio = torch.exp(logp - b_logp[idx])
                    pg = -torch.min(ratio * b_adv[idx], torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * b_adv[idx]).mean()
                    vl = 0.5 * (value(b_obs[idx]) - b_ret[idx]).square().mean()
                    loss = pg + 0.5 * vl
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value.parameters()) + [log_std], 0.5)
                    opt.step()
            rec = {"update": update, "stage": stage, "env_steps": env_steps, "physics_steps": env_steps * env.period, "mean_reward": float(torch.stack(buf["rew"]).mean()),
                   "std": float(std.mean()), "elapsed_s": time.time() - t0}
            if (update + 1) % args.val_every == 0 or update + 1 == total_updates:
                # export: fold the observation normalization into the policy's first layer
                export = Policy(cfg); export.load_state_dict(policy.state_dict())
                with torch.no_grad():
                    w = export.trunk[0].weight; b = export.trunk[0].bias
                    scale = 1.0 / torch.sqrt(obs_var + 1e-8)
                    b.copy_(b - w @ (obs_mean * scale)); w.copy_(w * scale)
                save_policy(out / "last.pt", export, {"update": update})
                val = validate(out / "last.pt", args.val_cases)
                rec["val"] = val
                if val["mean"] < best["val"]:
                    best = {"val": val["mean"], "update": update}
                    save_policy(out / "best.pt", export, {"update": update, "val": val})
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            msg = f"[ppo {args.kind} s{args.seed}] upd {update:3d} st {stage} reward {rec['mean_reward']:.3f} std {rec['std']:.3f} steps {env_steps} ({rec['elapsed_s']:.0f}s)"
            if "val" in rec:
                msg += f" | val {val['mean']:.3f} " + " ".join(f"{k[:4]}={v:.2f}" for k, v in val["per_regime"].items())
            print(msg, flush=True)
            update += 1
    (out / "best.json").write_text(json.dumps(best))
    print(f"done; best val {best['val']:.3f} at update {best['update']}; wall {time.time() - t0:.0f}s; env steps {env_steps}")


if __name__ == "__main__":
    main()
