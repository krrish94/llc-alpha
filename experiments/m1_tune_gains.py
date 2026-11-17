"""Milestone 1: tune PD + gravity-compensation gains with CMA-ES on the 'tune' case split.

    uv run python experiments/m1_tune_gains.py --target shared        # one gain set for all regimes
    uv run python experiments/m1_tune_gains.py --target contact       # gains for one regime
    uv run python experiments/m1_tune_gains.py --target ct_smooth     # computed-torque gains, smooth tracking

Protocol v1: 8 parameters (log stiffness per joint, log damping ratio), objective = mean composite
over the tune cases of the target regimes. Budget: popsize x iterations evaluations, logged.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cma
import numpy as np

from llc.controllers.pd import gains_from_params, home_inertia_diag
from llc.eval.cases import REGIMES
from llc.eval.parallel import CaseSpec, evaluate, mean_composite
from llc.sim.nominal import NominalModel

RUN_DIR = Path("runs/m1")
X0_PD = np.concatenate([np.log([400.0, 400.0, 300.0, 300.0, 100.0, 60.0, 30.0]), [0.0]])
LO_PD = np.concatenate([np.log([5.0] * 4 + [2.0] * 3), [np.log(0.2)]])
HI_PD = np.concatenate([np.log([3000.0] * 4 + [600.0] * 3), [np.log(4.0)]])
X0_CT = np.concatenate([np.log([20.0] * 7), [0.0]])
LO_CT = np.concatenate([np.log([2.0] * 7), [np.log(0.2)]])
HI_CT = np.concatenate([np.log([150.0] * 7), [np.log(4.0)]])


def pd_spec(params: np.ndarray, inertia_diag: np.ndarray) -> dict:
    kp, kd = gains_from_params(params, inertia_diag)
    return {"kind": "pd_gravity", "kp": kp.tolist(), "kd": kd.tolist()}


def ct_spec(params: np.ndarray) -> dict:
    return {"kind": "computed_torque", "omega": np.exp(params[:7]).tolist(), "zeta": float(np.exp(params[7]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--n-cases", type=int, default=6, help="tune cases per regime")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", default="v1", help="v2: deep-dive regimes with robot instances (ADR 0005)")
    ap.add_argument("--out-dir", default="runs/m1")
    ap.add_argument("--objective", default="composite", choices=["composite", "composite_cap"], help="composite_cap: the cap composite (protocol v3 blocked and task composites included; ADR 0010)")
    args = ap.parse_args()
    objective = (lambda res: float(np.mean([r[args.objective] for r in res])))  # the tuning objective (ADR 0010)
    global RUN_DIR
    RUN_DIR = Path(args.out_dir)

    computed_torque = args.target.startswith("ct_")
    regime_name = args.target[3:] if computed_torque else args.target
    if args.protocol == "v2":
        from llc.eval.cases_v2 import REGIMES_V2 as regime_list
    else:
        regime_list = REGIMES
    regimes = list(regime_list) if regime_name == "shared" else [regime_name]
    cases = [CaseSpec(r, "tune", i, protocol=args.protocol) for r in regimes for i in range(args.n_cases)]
    inertia_diag = home_inertia_diag(NominalModel())
    make_spec = (lambda x: ct_spec(x)) if computed_torque else (lambda x: pd_spec(x, inertia_diag))
    x0, lo, hi = (X0_CT, LO_CT, HI_CT) if computed_torque else (X0_PD, LO_PD, HI_PD)

    es = cma.CMAEvolutionStrategy(x0, 0.6, {"popsize": args.popsize, "bounds": [lo.tolist(), hi.tolist()], "seed": args.seed + 1, "verbose": -9})
    history, n_eval, t0 = [], 0, time.time()
    best_x, best_f = x0, np.inf
    for it in range(args.iters):
        xs = es.ask()
        specs = {f"c{i}": make_spec(np.asarray(x)) for i, x in enumerate(xs)}
        results = evaluate(specs, cases)
        fs = [objective(results[f"c{i}"]) for i in range(len(xs))]
        n_eval += len(xs) * len(cases)
        es.tell(xs, fs)
        i_best = int(np.argmin(fs))
        if fs[i_best] < best_f:
            best_f, best_x = fs[i_best], np.asarray(xs[i_best])
        history.append({"iter": it, "best": float(best_f), "pop_min": float(fs[i_best]), "pop_mean": float(np.mean(fs)), "elapsed_s": time.time() - t0})
        print(f"[{args.target}] iter {it:3d} pop min {fs[i_best]:.3f} mean {np.mean(fs):.3f} best {best_f:.3f} ({time.time() - t0:.0f}s)", flush=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "target": args.target, "regimes": regimes, "tune_cases": [c.id for c in cases], "params": best_x.tolist(),
        "spec": make_spec(best_x), "objective": float(best_f), "x0_objective": None,
        "episodes_evaluated": n_eval, "iters": args.iters, "popsize": args.popsize, "wall_s": time.time() - t0,
        "history": history, "inertia_diag_home": inertia_diag.tolist(),
    }
    out["x0_objective"] = objective(evaluate({"x0": make_spec(x0)}, cases)["x0"])
    out["objective_key"] = args.objective
    (RUN_DIR / f"gains_{args.target}.json").write_text(json.dumps(out, indent=1))
    print(f"[{args.target}] done: objective {best_f:.3f} (initial guess {out['x0_objective']:.3f}), {n_eval} episodes, {out['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
