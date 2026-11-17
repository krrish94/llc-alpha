"""Milestone 1: evaluate the conventional controllers on the test split and store metrics and
representative episodes.

    uv run python experiments/m1_evaluate.py --n-cases 16

Controllers: pd_initial (hand-set initial guess), pd_shared (one CMA-ES-tuned gain set for all
regimes), pd_tuned (the gain set tuned for each regime, evaluated on that regime), computed_torque
(privileged: full nominal inverse dynamics and the reference acceleration). Representative cases
per regime are chosen by a declared rule: the case whose pd_shared composite is the median.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from llc.eval.cases import REGIMES
from llc.eval.parallel import CaseSpec, evaluate

RUN_DIR = Path("runs/m1")


def load_gains(name: str) -> dict:
    return json.load(open(RUN_DIR / f"gains_{name}.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=16)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    shared, ct = load_gains("shared"), load_gains("ct_smooth")
    per_regime = {r: load_gains(r) for r in REGIMES}
    x0 = shared["params"]  # not used; the initial guess is reconstructed from the tuning script's X0
    from experiments.m1_tune_gains import X0_PD, pd_spec

    initial_spec = pd_spec(X0_PD, np.asarray(shared["inertia_diag_home"]))
    cases = [CaseSpec(r, args.split, i) for r in REGIMES for i in range(args.n_cases)]
    specs = {"pd_initial": initial_spec, "pd_shared": shared["spec"], "computed_torque": ct["spec"]}
    results = evaluate(specs, cases)
    results["pd_tuned"] = []
    for r in REGIMES:
        res_r = evaluate({"pd_tuned": per_regime[r]["spec"]}, [c for c in cases if c.regime == r])["pd_tuned"]
        results["pd_tuned"].extend(res_r)
    # representative cases: median pd_shared composite per regime
    chosen = {}
    for r in REGIMES:
        rows = [m for m in results["pd_shared"] if m["regime"] == r]
        order = sorted(rows, key=lambda m: m["composite"])
        chosen[r] = order[len(order) // 2]["case_id"]
    rep_cases = [c for c in cases if c.id in chosen.values()]
    episodes = {}
    for name, spec in specs.items():
        res = evaluate({name: spec}, rep_cases, keep_episodes=True)[name]
        for m in res:
            episodes[(name, m["case_id"])] = m.pop("episode")
    for r in REGIMES:
        res = evaluate({"pd_tuned": per_regime[r]["spec"]}, [c for c in rep_cases if c.regime == r], keep_episodes=True)["pd_tuned"]
        for m in res:
            episodes[("pd_tuned", m["case_id"])] = m.pop("episode")
    ep_dir = RUN_DIR / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    for (name, cid), ep in episodes.items():
        np.savez_compressed(ep_dir / f"{cid}__{name}.npz", **{k: getattr(ep, k) for k in ("t", "q", "qd", "q_ref", "qd_ref", "tau", "tool_pos", "f_normal", "f_tangent", "n_tool_contacts", "n_other_contacts")})
    git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    out = {
        "split": args.split, "n_cases_per_regime": args.n_cases, "git": git,
        "controllers": {"pd_initial": initial_spec, "pd_shared": shared["spec"], "computed_torque": ct["spec"],
                        "pd_tuned": {r: per_regime[r]["spec"] for r in REGIMES}},
        "tuning": {name: {"objective": g["objective"], "x0_objective": g["x0_objective"], "episodes_evaluated": g["episodes_evaluated"], "wall_s": g["wall_s"], "params": g["params"]}
                   for name, g in [("shared", shared), ("ct_smooth", ct)] + [(r, per_regime[r]) for r in REGIMES]},
        "representative_cases": chosen,
        "results": results,
    }
    (RUN_DIR / f"metrics_{args.split}.json").write_text(json.dumps(out, indent=1))
    for name in results:
        print(f"{name:16s}", " ".join(f"{r[:5]}={np.mean([m['composite'] for m in results[name] if m['regime']==r]):6.2f}" for r in REGIMES))


if __name__ == "__main__":
    main()
