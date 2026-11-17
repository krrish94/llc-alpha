"""Reward decomposition in the training medium: per-term mean step reward of a zero-residual policy
(the PD base) and of trained checkpoints on identical windows of each curriculum stage. Used to tell
an optimizer failure (trained policy scores below its initialization) from a reward
mis-specification (trained policy scores higher by sacrificing tracking).

    uv run python experiments/d_reward_terms.py --out runs/d3/reward_terms.json acmed=runs/d3/pd_residual_acmed-s0/best.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from experiments.d2_train import STAGES
from llc.controllers.pd import home_inertia_diag
from llc.dynamics.rigid_body import RigidBodyModel
from llc.learning.actor_critic import Critic, RewardConfig, rollout_ac
from llc.learning.data import reference_pool_v2, sample_batch_v2
from llc.learning.mjgrad import MjBatch
from llc.learning.policy import Policy, PolicyConfig, load_policy
from llc.sim.instances import sample_instance
from llc.sim.nominal import NominalModel
from llc.sim.panda import load_model


def reward_terms(rew: RewardConfig, q_ref, qd_ref, res: dict, lim, dt: float) -> dict[str, torch.Tensor]:
    """Per-step reward terms (B, T-1) recomputed from the rollout logs (states after each step)."""
    B, T = res["q"].shape[:2]
    q, qd, tau, touch = res["q"], res["qd"], res["tau"], res["touch"]
    if getattr(rew, "force_tau", 0.0) > 0:  # the reward's force terms see the low-pass filtered force (ADR 0007)
        a, f = min(1.0, dt / rew.force_tau), torch.zeros_like(touch)
        acc = torch.zeros(touch.shape[0])
        for k in range(touch.shape[1]):
            acc = acc + a * (touch[:, k] - acc); f[:, k] = acc
        touch = f
    e, ed = q_ref[:, 1:] - q[:, :-1], qd_ref[:, 1:] - qd[:, :-1]
    last_tau = torch.cat([torch.zeros(B, 1, tau.shape[-1]), tau[:, :-1]], 1)[:, :-1]
    prev_touch = torch.cat([torch.zeros(B, 1), touch[:, :-1]], 1)[:, :-1]
    tau, touch = tau[:, :-1], touch[:, :-1]
    return {
        "position kernel": rew.w_pos * torch.exp(-(e * e).mean(-1) / rew.pos_sigma**2),
        "velocity kernel": rew.w_vel * torch.exp(-(ed * ed).mean(-1) / rew.vel_sigma**2),
        "torque": -rew.w_torque * (tau / lim).square().sum(-1),
        "torque rate": -rew.w_rate * ((tau - last_tau) / (dt * rew.rate_scale)).square().sum(-1),
        "contact force": -rew.w_force * (torch.relu(touch - rew.force_soft) / rew.force_scale).square(),
        "force rate": -rew.w_impulse * ((touch - prev_touch) / (dt * rew.impulse_scale)).square(),
        "linear restoring": -rew.w_lin * (torch.sqrt(e * e + 1e-4).mean(-1) / rew.lin_scale),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", help="name=path pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gains", default="runs/d1/gains_shared.json")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--pos-sigma", type=float, default=0.05)
    ap.add_argument("--w-rate", type=float, default=None)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    inertia = home_inertia_diag(NominalModel())
    shared = json.load(open(args.gains))["spec"]
    cfg = PolicyConfig(kind="pd_residual", history=10, history_stride=10, history_dense=32, rate_hz=1000.0, latency_s=0.0,
                       inertia_diag=tuple(inertia.tolist()), kp_shared=tuple(shared["kp"]), kd_shared=tuple(shared["kd"]))
    policies = {"pd_base": Policy(cfg)}  # zero residual by initialization
    for item in args.checkpoints:
        name, path = item.split("=")
        policies[name], _ = load_policy(path)
    gravity_model = RigidBodyModel(load_model(), dtype=torch.float32)
    pool = reference_pool_v2(48, "train")
    B, T, dt = args.batch, 2000, 1e-3
    batch = MjBatch([sample_instance(rng, payload_mass=float(rng.uniform(0, 3))) for _ in range(B)], seed=1)
    critic = Critic(cfg.obs_dim, 4)
    rew = RewardConfig(pos_sigma=args.pos_sigma)
    if args.w_rate is not None:
        rew.w_rate = args.w_rate
    lim = policies["pd_base"].limits
    out = {"reward": rew.__dict__, "stages": []}
    for stage, mix in enumerate(STAGES):
        q_ref, qd_ref, q0, qd0, labels, f_ext, _priv = sample_batch_v2(pool, mix, B, T, dt, rng)
        payloads = np.array([float(rng.uniform(0, 3)) for _ in range(B)], np.float32)
        perts = [sample_instance(rng, payload_mass=float(payloads[i]), payload_offset=(rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.10)) for i in range(B)]
        q_ref_t, qd_ref_t = torch.as_tensor(q_ref), torch.as_tensor(qd_ref)
        rec = {"stage": stage, "labels": {str(l): int(c) for l, c in zip(*np.unique(labels, return_counts=True))}, "policies": {}}
        for name, pol in policies.items():
            batch.perturb(perts)
            with torch.no_grad():
                res = rollout_ac(pol, critic, batch, gravity_model, q_ref_t, qd_ref_t, q0, qd0, payloads, f_ext, rew, horizon=100, train=False, bootstrap_scale=0.0)
            terms = reward_terms(rew, q_ref_t, qd_ref_t, res, lim, dt)
            lab = np.asarray(labels)
            rec["policies"][name] = {"terms": {k: float(v.mean()) for k, v in terms.items()}, "total": float(sum(v.mean() for v in terms.values())),
                                     "position_kernel_by_family": {str(l): float(terms["position kernel"][lab == l].mean()) for l in sorted(set(labels))}}
            print(f"stage {stage} {name:14s} total {rec['policies'][name]['total']:7.4f} | " + " ".join(f"{k}={v:7.4f}" for k, v in rec["policies"][name]["terms"].items()), flush=True)
        out["stages"].append(rec)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
