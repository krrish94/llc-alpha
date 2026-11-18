"""Where the blocked family's excess force comes from (ADR 0011 amendment 3, item 2): the transient after the impact or
the steady press? Runs the controllers on the in-distribution blocked test cases through a profile, keeps the episodes
and splits the blocked phase of each into its first 0.5 s, its middle and its last 0.5 s: mean normal force, peak, and
the share of the excess above the cap that falls in each part; also the time from the impact to the force first
settling under the cap for 0.2 s.

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/m11_blocked_transient.py --profile p1 --runs student_p1=runs/m11/student_p1-s0/best.pt --out runs/m11/blocked_transient_p1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from llc.eval.cases_v2 import make_case_v2
from llc.eval.parallel import CaseSpec, evaluate
from llc.sim.nominal import NominalModel
from llc.sim.plant import PLANT


def split_stats(f: np.ndarray, t: np.ndarray, mask: np.ndarray, cap: float) -> dict:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {}
    t0, t1 = t[idx[0]], t[idx[-1]]
    parts = {"first_0.5s": mask & (t < t0 + 0.5), "middle": mask & (t >= t0 + 0.5) & (t <= t1 - 0.5), "last_0.5s": mask & (t > t1 - 0.5)}
    exc = np.maximum(f - cap, 0.0)
    total = float(exc[mask].sum()) + 1e-9
    out = {"peak": float(f[mask].max()), "mean": float(f[mask].mean()), "excess_mean": float(exc[mask].mean())}
    for k, m in parts.items():
        out[f"{k}_mean"] = float(f[m].mean()) if m.any() else float("nan")
        out[f"{k}_excess_share"] = float(exc[m].sum() / total)
    under = (f[idx] <= cap)
    settle = None
    for i in range(len(idx)):
        j = i + 200
        if j <= len(idx) and under[i:j].all():
            settle = float(t[idx[i]] - t0); break
    out["settle_under_cap_s"] = settle if settle is not None else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="")
    ap.add_argument("--gains", default="runs/m9/gains_shared.json")
    ap.add_argument("--runs", nargs="*", default=[], help="name=checkpoint pairs")
    ap.add_argument("--cases", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gains = json.load(open(args.gains))["spec"]
    wrap = (lambda s: {"kind": "servo", "profile": args.profile, "inner": s, "interface": "model", "kp": gains["kp"], "kd": gains["kd"]}) if args.profile else (lambda s: s)
    specs = {"pd_feedforward": wrap({**gains, "kind": "pd_feedforward"}), "oracle_pd": wrap({"kind": "oracle", "base": {**gains, "kind": "pd_feedforward"}})}
    for item in args.runs:
        name, path = item.split("=")
        specs[name] = wrap({"kind": "learned", "path": path})
    cases = [CaseSpec("blocked", "test", i, protocol="v2") for i in range(args.cases)]
    res = evaluate(specs, cases, keep_episodes=True, workers=args.workers)
    nominal = NominalModel()
    out = {}
    for name, rows in res.items():
        per = []
        for spec, m in zip(cases, rows):
            case = make_case_v2(spec.regime, spec.split, spec.index, nominal)
            ep = m["episode"]; ref = case.reference; n = len(ep.t)
            s = split_stats(ep.f_normal, ep.t, ref.phase_mask("blocked")[:n], PLANT.force_cap)
            s["payload"] = float(case.perturbation.payload_mass); s["case"] = spec.index
            per.append(s)
        agg = {k: float(np.nanmean([p[k] for p in per if k in p])) for k in per[0] if k not in ("case",)}
        agg["n"] = len(per)
        for pl in sorted({p["payload"] for p in per}):
            sub = [p for p in per if p["payload"] == pl]
            agg[f"excess_mean_payload_{pl:g}"] = float(np.nanmean([p["excess_mean"] for p in sub]))
            agg[f"first_0.5s_mean_payload_{pl:g}"] = float(np.nanmean([p["first_0.5s_mean"] for p in sub]))
            agg[f"last_0.5s_mean_payload_{pl:g}"] = float(np.nanmean([p["last_0.5s_mean"] for p in sub]))
        out[name] = {"aggregate": agg, "per_case": per}
        print(f"{name:18s} blocked phase: mean {agg['mean']:.1f} N, peak {agg['peak']:.0f}; first 0.5 s {agg['first_0.5s_mean']:.1f}, middle {agg['middle_mean']:.1f}, last 0.5 s {agg['last_0.5s_mean']:.1f}; "
              f"excess {agg['excess_mean']:.1f} N of which first 0.5 s {100 * agg['first_0.5s_excess_share']:.0f}%, middle {100 * agg['middle_excess_share']:.0f}%, last {100 * agg['last_0.5s_excess_share']:.0f}%; settle under the cap {agg['settle_under_cap_s']:.2f} s; "
              + " ".join(f"excess@{pl:g}kg {agg[f'excess_mean_payload_{pl:g}']:.1f}" for pl in sorted({p['payload'] for p in per})), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
