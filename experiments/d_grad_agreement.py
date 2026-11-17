"""Gradient agreement diagnostic (gradient plan, item 2): along the same plant trajectories, compare
the gradient of a k-step return-like loss with respect to the policy's torques obtained (a) by
finite differences through MuJoCo and (b) through the learned dynamics model (decoupled backward),
on free-space and in-contact windows. Reports cosine similarity and norm ratio per horizon.

    uv run python experiments/d_grad_agreement.py --run runs/d3/pd_residual_dec-s0 --out runs/d3/grad_agreement.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from llc.dynamics.rigid_body import RigidBodyModel
from llc.eval.cases_v2 import make_case_v2
from llc.learning.decoupled import DecoupledBatch, DynamicsNet
from llc.learning.mjgrad import MjBatch
from llc.learning.policy import load_policy
from llc.sim.nominal import NominalModel
from llc.sim.panda import load_model


def window_gradient(policy, batch, gravity_model, q_ref, qd_ref, q0, qd0, horizon):
    """Gradient of sum_t |e_t|^2 + (touch_t/50)^2 over the window w.r.t. a torque offset parameter."""
    from llc.learning.policy import HistoryBuffer

    theta = torch.zeros(q0.shape[0], 7, requires_grad=True)
    batch.reset(q0, qd0)
    q, qd = (x.float() for x in batch.state())
    buf = HistoryBuffer(policy.cfg, q, qd)
    last_tau = torch.zeros_like(q)
    loss = torch.zeros(())
    with torch.enable_grad():
        for k in range(horizon):
            buf.push(q, qd, last_tau, q_ref[:, k])
            obs = buf.observation(q, qd, q_ref[:, k], qd_ref[:, k], policy.limits)
            held = policy.decode(policy(obs))
            tau = policy.fast_torque(held, q, qd, q_ref[:, k], qd_ref[:, k], gravity_model.bias(q, torch.zeros_like(q))) + theta
            tau = torch.maximum(torch.minimum(tau, policy.limits), -policy.limits)
            q, qd, touch, _ = batch.step(q, qd, tau, None)
            q, qd, touch = q.float(), qd.float(), touch.float()
            loss = loss + ((q_ref[:, k + 1] - q).square().sum(-1) / 0.02**2 + (touch / 50.0).square()).sum()
            last_tau = tau
        loss.backward()
    return theta.grad.detach().clone()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory with best.pt and dynamics_model.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizons", type=int, nargs="*", default=[1, 8, 32, 100])
    ap.add_argument("--n-cases", type=int, default=6)
    args = ap.parse_args()
    policy, _ = load_policy(Path(args.run) / "best.pt")
    net = DynamicsNet(); net.load_state_dict(torch.load(Path(args.run) / "dynamics_model.pt")); net.eval()
    nominal = NominalModel()
    gravity_model = RigidBodyModel(load_model(), dtype=torch.float32)
    results = {}
    for regime, start_kind in (("free", "random"), ("contact2", "contact")):
        cases = [make_case_v2(regime, "val", i, nominal) for i in range(args.n_cases)]
        for H in args.horizons:
            cos, ratio = [], []
            for case in cases:
                ref = case.reference
                k_c = int(round(case.meta.get("t_contact_nominal", 0.3) / 1e-3))
                k0 = max(0, k_c - 20) if start_kind == "contact" else int(len(ref.t) * 0.3)
                if k0 + H + 1 >= len(ref.t):
                    continue
                q_ref = torch.tensor(ref.q[k0:k0 + H + 1], dtype=torch.float32)[None]
                qd_ref = torch.tensor(ref.qd[k0:k0 + H + 1], dtype=torch.float32)[None]
                q0, qd0 = ref.q[k0][None], ref.qd[k0][None]
                fd = MjBatch([case.perturbation], seed=0)
                g_fd = window_gradient(policy, fd, gravity_model, q_ref, qd_ref, q0, qd0, H)
                dec = DecoupledBatch(MjBatch([case.perturbation], seed=0), net, np.random.default_rng(0))
                dec.perturb([case.perturbation]); dec.record = False
                g_m = window_gradient(policy, dec, gravity_model, q_ref, qd_ref, q0, qd0, H)
                a, b = g_m.flatten(), g_fd.flatten()
                cos.append(float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)))
                ratio.append(float(a.norm() / (b.norm() + 1e-12)))
            results[f"{regime}/H{H}"] = {"cosine_median": float(np.median(cos)), "cosine_min": float(np.min(cos)), "norm_ratio_median": float(np.median(ratio)), "n": len(cos)}
            print(f"{regime:9s} H={H:3d}: cosine median {np.median(cos):.3f} min {np.min(cos):.3f}; |model|/|fd| median {np.median(ratio):.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
