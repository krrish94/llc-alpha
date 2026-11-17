"""Milestone 6: multi-phase trajectories (free motion, contact onset, sustained pressing and
sliding, release) at several speeds and payloads; learned controllers against fixed gains and
phase-scheduled gains with switching/interpolation (oracle phase labels from the reference).

    uv run python experiments/m6_phases.py --runs runs/m2/... --out runs/m6/phases.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from llc.eval.parallel import CaseSpec, evaluate
from llc.sim.panda import DynamicsPerturbation

SPEEDS = [1.0, 1.5, 2.0]
PAYLOADS = [0.0, 1.0]
BLENDS = [0.0, 0.05, 0.2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--n-cases", type=int, default=8)
    ap.add_argument("--out", default="runs/m6/phases.json")
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()
    m1 = json.load(open("runs/m1/metrics_test.json"))
    tuned = m1["controllers"]["pd_tuned"]
    phase_gains = {"default": tuned["transition"], "approach": tuned["transition"], "release": tuned["transition"], "free": tuned["transition"],
                   "press": tuned["contact"], "slide": tuned["contact"]}
    specs = {"pd_shared": m1["controllers"]["pd_shared"], "pd_contact_gains": tuned["contact"], "pd_transition_gains": tuned["transition"]}
    for b in BLENDS:
        specs[f"pd_phase_blend{int(b*1000):03d}ms"] = {"kind": "pd_phase", "phase_gains": phase_gains, "blend": b}
    for run in args.runs:
        run = Path(run)
        specs[run.name] = {"kind": "learned", "path": str(run / args.ckpt)}
    grid = {}
    for regime in ("contact", "transition"):
        for speed in SPEEDS:
            for payload in PAYLOADS:
                pert = DynamicsPerturbation(payload_mass=payload, payload_offset=(0.0, 0.0, 0.10))
                cases = [CaseSpec(regime, "test", i, perturbation=pert, speed_scale=speed) for i in range(args.n_cases)]
                key = f"{regime}|speed{speed}|payload{payload}"
                grid[key] = evaluate(specs, cases)
                print(key + ": " + " ".join(f"{n}={np.mean([m['composite'] for m in r]):.2f}" for n, r in grid[key].items()), flush=True)
    git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"git": git, "speeds": SPEEDS, "payloads": PAYLOADS, "blends": BLENDS, "n_cases": args.n_cases, "controllers": specs, "grid": grid}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
