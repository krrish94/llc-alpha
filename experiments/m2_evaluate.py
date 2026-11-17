"""Milestones 2 and 3: evaluate learned controllers and baselines on the test split, in MuJoCo and
(optionally) in the torch model, including out-of-distribution speed scales.

    uv run python experiments/m2_evaluate.py --runs runs/m2/learned_gains-s0 runs/m2/pd_residual-s0 ... --out runs/m2/metrics_test.json

Each run directory contributes its best.pt under the name <kind>[-<tag>] with the seed recorded,
so seeds can be pooled or shown separately in the report.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from llc.eval.cases import REGIMES, make_case
from llc.eval.metrics import compute_metrics
from llc.eval.parallel import CaseSpec, evaluate
from llc.eval.rollout import Episode
from llc.sim.nominal import NominalModel
from llc.sim.panda import DynamicsPerturbation


def model_rollout_episode(policy, case, dt_model=0.0025, contact=None) -> Episode:
    """Closed loop in the torch model (not the plant), logged in the Episode format for the same metrics."""
    from llc.dynamics.contact import ContactParams
    from llc.dynamics.rigid_body import RigidBodyModel
    from llc.learning.rollout import RigidBodyDynamics, RolloutConfig, rollout
    from llc.sim.panda import load_model

    model = RigidBodyModel(load_model(), dtype=torch.float32)
    dyn = RigidBodyDynamics(model, contact or ContactParams(k_n=40000.0, c_n=400.0, mu=0.5))
    ref = case.reference
    t = np.arange(0.0, ref.duration + 1e-9, dt_model)
    rs = lambda x: torch.tensor(np.stack([np.interp(t, ref.t, x[:, j]) for j in range(7)], 1), dtype=torch.float32)[None]
    q0 = torch.tensor(case.q0, dtype=torch.float32)[None]
    f_ext = None
    if case.pulse is not None:
        f_ext = torch.zeros(1, len(t), 3)
        on = (t >= case.pulse.t_start) & (t < case.pulse.t_start + case.pulse.duration)
        f_ext[0, torch.as_tensor(on)] = torch.tensor(case.pulse.force, dtype=torch.float32)
    with torch.no_grad():
        res = rollout(policy, dyn, rs(ref.q), rs(ref.qd), q0, torch.zeros_like(q0), RolloutConfig(dt=dt_model, horizon_s=t[-1] + dt_model, tbptt_steps=0), f_ext=f_ext)
    n = len(t)
    q, qd, tau, f, p = (res[k][0, :n].numpy() for k in ("q", "qd", "tau", "f", "tool_pos"))
    f_mag = np.linalg.norm(f, axis=1)
    # interpolate back to the 1 kHz grid so phase masks and metrics apply unchanged
    up = lambda x: np.stack([np.interp(ref.t, t, x[:, j]) for j in range(x.shape[1])], 1) if x.ndim == 2 else np.interp(ref.t, t, x)
    f_up = up(f_mag)
    return Episode(case.id, case.regime, ref.t, up(q), up(qd), ref.q, ref.qd, up(tau), up(tau), up(p), f_up, np.zeros_like(f_up),
                   (f_up > 0.5).astype(int), np.zeros(len(ref.t), int), bool(not np.all(np.isfinite(q))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--n-cases", type=int, default=16)
    ap.add_argument("--out", default="runs/m2/metrics_test.json")
    ap.add_argument("--ood-scales", type=float, nargs="*", default=[1.5, 2.0])
    ap.add_argument("--in-model", action="store_true", help="also evaluate learned controllers inside the torch model")
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()
    m1 = json.load(open("runs/m1/metrics_test.json"))
    specs = {"pd_shared": m1["controllers"]["pd_shared"], "computed_torque": m1["controllers"]["computed_torque"]}
    learned = {}
    for run in args.runs:
        run = Path(run)
        meta = json.load(open(run / "meta.json"))
        name = run.name  # e.g. learned_gains-s0
        learned[name] = {"kind": "learned", "path": str(run / args.ckpt), "policy_kind": meta["policy_config"]["kind"], "seed": meta["args"]["seed"]}
        specs[name] = {"kind": "learned", "path": str(run / args.ckpt)}
    cases = [CaseSpec(r, "test", i) for r in REGIMES for i in range(args.n_cases)]
    results = evaluate(specs, cases)
    per_regime_tuned = m1["controllers"]["pd_tuned"]
    results["pd_tuned"] = []
    for r in REGIMES:
        results["pd_tuned"].extend(evaluate({"pd_tuned": per_regime_tuned[r]}, [c for c in cases if c.regime == r])["pd_tuned"])
    ood = {}
    for s in args.ood_scales:
        ood_cases = [CaseSpec(r, "test", i, speed_scale=s) for r in ("smooth", "zoh", "step", "contact", "transition") for i in range(args.n_cases // 2)]
        ood[str(s)] = evaluate(specs, ood_cases)
    in_model = {}
    if args.in_model:
        from llc.learning.policy import load_policy

        nominal = NominalModel()
        for name, info in learned.items():
            policy, _ = load_policy(info["path"])
            in_model[name] = []
            for c in cases:
                case = make_case(c.regime, c.split, c.index, nominal)
                ep = model_rollout_episode(policy, case)
                m = compute_metrics(ep, case)
                m["case_id"], m["regime"] = case.id, case.regime
                in_model[name].append(m)
    git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    out = {"git": git, "n_cases_per_regime": args.n_cases, "controllers": specs, "learned": learned, "results": results, "ood": ood, "in_model": in_model}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for name in results:
        print(f"{name:22s}", " ".join(f"{r[:5]}={np.mean([m['composite'] for m in results[name] if m['regime']==r]):6.2f}" for r in REGIMES),
              " div", int(sum(m["diverged"] for m in results[name])))


if __name__ == "__main__":
    main()
