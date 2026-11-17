"""Milestone 3: robustness of controllers to changed plant dynamics (one factor at a time, plus
random combinations), evaluated in MuJoCo on the test split.

    uv run python experiments/m3_robustness.py --runs runs/m2/... --out runs/m3/robustness.json

Factors and levels (protocol v1): payload mass (kg) at 10 cm along the tool, payload centre of mass
offset (m, lateral), inertia scale, joint dry friction (Nm), actuation delay (s), observation noise
(rad, rad/s) and reference speed scale. Levels marked as held-out in the report are outside the
randomization range used for DR training (ADR 0004).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from llc.eval.cases import REGIMES
from llc.eval.parallel import CaseSpec, evaluate
from llc.sim.panda import DynamicsPerturbation

FACTORS = {
    "payload_mass": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "payload_lateral": [0.0, 0.03, 0.06, 0.1],
    "inertia_scale": [0.7, 1.0, 1.3, 1.6],
    "friction_loss": [0.0, 0.5, 1.0, 2.0],
    "actuation_delay": [0.0, 0.005, 0.010, 0.020],
    "obs_noise": [0.0, 0.001, 0.003, 0.01],
    "speed_scale": [0.5, 1.0, 1.5, 2.0],
}


def perturbation(factor: str, level: float) -> tuple[DynamicsPerturbation, float]:
    """(plant perturbation, reference speed scale) for a factor level."""
    if factor == "payload_mass":
        return DynamicsPerturbation(payload_mass=level, payload_offset=(0.0, 0.0, 0.10)), 1.0
    if factor == "payload_lateral":
        return DynamicsPerturbation(payload_mass=1.0, payload_offset=(level, 0.0, 0.10)), 1.0
    if factor == "inertia_scale":
        return DynamicsPerturbation(inertia_scale=level), 1.0
    if factor == "friction_loss":
        return DynamicsPerturbation(friction_loss=level), 1.0
    if factor == "actuation_delay":
        return DynamicsPerturbation(actuation_delay=level), 1.0
    if factor == "obs_noise":
        return DynamicsPerturbation(obs_noise_q=level, obs_noise_qd=10.0 * level), 1.0
    if factor == "speed_scale":
        return DynamicsPerturbation(), level
    raise ValueError(factor)


def random_combination(rng: np.random.Generator) -> tuple[DynamicsPerturbation, float, dict]:
    vals = {"payload_mass": float(rng.uniform(0, 2.5)), "lateral": float(rng.uniform(0, 0.08)), "inertia_scale": float(rng.uniform(0.7, 1.5)),
            "friction_loss": float(rng.uniform(0, 1.5)), "actuation_delay": float(rng.choice([0.0, 0.005, 0.01, 0.015])),
            "obs_noise": float(rng.choice([0.0, 0.001, 0.003])), "speed_scale": float(rng.uniform(0.6, 1.8))}
    p = DynamicsPerturbation(payload_mass=vals["payload_mass"], payload_offset=(vals["lateral"], 0.0, 0.10), inertia_scale=vals["inertia_scale"],
                             friction_loss=vals["friction_loss"], actuation_delay=vals["actuation_delay"], obs_noise_q=vals["obs_noise"], obs_noise_qd=10 * vals["obs_noise"])
    return p, vals["speed_scale"], vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--n-cases", type=int, default=6, help="test cases per regime and level")
    ap.add_argument("--n-combos", type=int, default=12)
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--out", default="runs/m3/robustness.json")
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()
    m1 = json.load(open("runs/m1/metrics_test.json"))
    specs = {"pd_shared": m1["controllers"]["pd_shared"]}
    for run in args.runs:
        run = Path(run)
        specs[run.name] = {"kind": "learned", "path": str(run / args.ckpt)}
    sweeps = {}
    for factor, levels in FACTORS.items():
        sweeps[factor] = {}
        for level in levels:
            pert, speed = perturbation(factor, level)
            cases = [CaseSpec(r, "test", i, perturbation=pert, speed_scale=speed) for r in args.regimes for i in range(args.n_cases)]
            sweeps[factor][str(level)] = evaluate(specs, cases)
            print(f"{factor} = {level}: " + " ".join(f"{n}={np.mean([m['composite'] for m in res]):.2f}" for n, res in sweeps[factor][str(level)].items()), flush=True)
    rng = np.random.default_rng(3)
    combos = []
    for i in range(args.n_combos):
        pert, speed, vals = random_combination(rng)
        cases = [CaseSpec(r, "test", i % args.n_cases, perturbation=pert, speed_scale=speed) for r in args.regimes]
        combos.append({"values": vals, "results": evaluate(specs, cases)})
    git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"git": git, "factors": FACTORS, "n_cases": args.n_cases, "regimes": args.regimes, "controllers": specs, "sweeps": sweeps, "combinations": combos}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
