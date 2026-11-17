"""Deep dive: evaluate controllers on protocol v2 (in-distribution families, held-out workspace,
speeds, payload and instance deviations).

    uv run python experiments/d_evaluate.py --gains runs/d1/gains_shared.json --runs runs/d2/... --out runs/d1/metrics_test.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from llc.eval.cases_v2 import REGIMES_V2
from llc.eval.parallel import CaseSpec, evaluate


def case_groups(n: int) -> dict[str, list[CaseSpec]]:
    g = {"in_distribution": [CaseSpec(r, "test", i, protocol="v2") for r in REGIMES_V2 for i in range(n)]}
    g["heldout_workspace"] = [CaseSpec(r, "test", i, protocol="v2", heldout_workspace=True) for r in ("free", "policy") for i in range(n // 2)]
    for s in (1.5, 2.0):
        g[f"speed_x{s:g}"] = [CaseSpec(r, "test", i, protocol="v2", speed_scale=s) for r in ("free", "policy", "contact2", "task") for i in range(n // 2)]
    g["payload_3kg"] = [CaseSpec(r, "test", i, protocol="v2", payload=3.0) for r in ("free", "contact2", "task", "blocked") for i in range(n // 2)]
    g["ood_instances"] = [CaseSpec(r, "test", i, protocol="v2", ood_instance=True) for r in ("free", "policy", "contact2", "blocked") for i in range(n // 2)]
    for k in (2000.0, 500.0):  # compliant table and wall (ADR 0008): does the controller behave as a spring on its error?
        g[f"surface_{k:g}"] = [CaseSpec(r, "test", i, protocol="v2", surface_stiffness=k) for r in ("contact2", "blocked", "task") for i in range(n // 2)]
    from llc.eval.cases_v2 import TASK_REGIMES

    if TASK_REGIMES:  # Milestone 9 task families (gripper plant)
        g["tasks"] = [CaseSpec(r, "test", i, protocol="v2") for r in TASK_REGIMES for i in range(n)]
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", default="runs/d1/gains_shared.json", help="protocol v2 shared PD gains")
    ap.add_argument("--extra-gains", nargs="*", default=["pd_shared_v1=runs/m1/gains_shared.json"], help="name=path of additional PD gain sets")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--n-cases", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--save-episodes", default="", help="directory: store episodes of the representative case per in-distribution regime (median pd_shared composite) for every controller")
    ap.add_argument("--feedforward", action="store_true", help="add the shared PD with the nominal inverse-dynamics feedforward of the reference (pd_feedforward)")
    ap.add_argument("--oracle-on", nargs="*", default=[], help="add the privileged contact-impedance oracle around these bases: 'pd' or a run directory (ADR 0008)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--latency", type=float, default=0.0, help="s; adds the group latency_<ms>ms: the in-distribution families with every learned controller at this compute latency (ADR 0010)")
    args = ap.parse_args()
    specs = {"pd_shared": json.load(open(args.gains))["spec"]}
    for item in args.extra_gains:
        name, path = item.split("=")
        specs[name] = json.load(open(path))["spec"]
    if args.feedforward:
        specs["pd_feedforward"] = {**specs["pd_shared"], "kind": "pd_feedforward"}
        for item in args.extra_gains:  # the feedforward variant of every extra PD gain set (ADR 0010 classical set)
            name = item.split("=")[0]
            if specs[name]["kind"] == "pd_gravity":
                specs[f"{name}_feedforward"] = {**specs[name], "kind": "pd_feedforward"}
    for base in args.oracle_on:
        if base == "pd":
            specs["oracle_pd"] = {"kind": "oracle", "base": specs["pd_shared"]}
        else:
            run, _, ckpt = base.partition("@")
            specs[f"oracle_{Path(run).name}"] = {"kind": "oracle", "base": {"kind": "learned", "path": str(Path(run) / (ckpt or args.ckpt))}}
    learned = {}
    for run in args.runs:
        run, _, ckpt = run.partition("@")  # "runs/d4/x-s0@final.pt" overrides --ckpt for that run
        run = Path(run)
        ckpt = ckpt or args.ckpt
        meta = json.load(open(run / "meta.json"))
        name = run.name if run.name not in specs else f"{run.parent.name}_{run.name}"  # runs of different milestones may share a directory name
        learned[name] = {"path": str(run / ckpt), "policy_kind": meta["policy_config"]["kind"], "seed": meta["args"]["seed"], "variant": meta["args"].get("variant", ""), "checkpoint": ckpt,
                         "method_key": meta.get("method_key")}  # explicit pooling key (PPO runs); kind_variant otherwise
        specs[name] = {"kind": "learned", "path": str(run / ckpt)}
    groups = case_groups(args.n_cases)
    if args.groups:
        groups = {k: v for k, v in groups.items() if k in args.groups}
    results = {}
    git = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    from llc.sim.plant import PLANT

    def save(representative: dict, partial: bool) -> None:  # every finished group is on disk before the next starts
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"git": git, "protocol": "v2", "plant": PLANT.name, "n_cases": args.n_cases, "controllers": specs, "learned": learned, "groups": results, "representative_cases": representative, "partial": partial}, open(args.out, "w"), indent=1)

    for gname, cases in groups.items():
        results[gname] = evaluate(specs, cases, workers=args.workers)
        save({}, True)
    if args.latency > 0:  # realism check: the learned controllers with a compute latency, same cases as the in-distribution group
        lat_specs = {name: ({**spec, "options": {**spec.get("options", {}), "latency_s": args.latency}} if spec["kind"] == "learned" else spec) for name, spec in specs.items()}
        gname = f"latency_{int(round(1000 * args.latency))}ms"
        groups[gname] = case_groups(args.n_cases)["in_distribution"]
        results[gname] = evaluate(lat_specs, groups[gname], workers=args.workers)
        save({}, True)
        print(f"== {gname}")
        for name, res in results[gname].items():
            order = list(REGIMES_V2) + ["push", "pick"]
            regs = sorted({m["regime"] for m in res}, key=lambda r: order.index(r) if r in order else len(order))
            print(f"   {name:24s}", " ".join(f"{r[:6]}={np.mean([m['composite'] for m in res if m['regime']==r]):5.2f}" for r in regs), " exc", f"{np.mean([m['excess_force'] for m in res]):5.1f}", " div", int(sum(m["diverged"] for m in res)), flush=True)
    representative = {}
    if args.save_episodes and "in_distribution" in results:
        ep_dir = Path(args.save_episodes)
        ep_dir.mkdir(parents=True, exist_ok=True)
        for r in REGIMES_V2:
            rows = sorted([m for m in results["in_distribution"]["pd_shared"] if m["regime"] == r], key=lambda m: m["composite"])
            if rows:
                representative[r] = rows[len(rows) // 2]["case_id"]
        rep_cases = [c for c in groups["in_distribution"] if c.id in representative.values()]
        for name, spec in specs.items():
            res = evaluate({name: spec}, rep_cases, keep_episodes=True, workers=args.workers)[name]
            for m in res:
                ep = m.pop("episode")
                np.savez_compressed(ep_dir / f"{m['case_id']}__{name}.npz", **{k: getattr(ep, k) for k in ("t", "q", "qd", "q_ref", "qd_ref", "tau", "tool_pos", "f_normal", "f_tangent", "n_tool_contacts", "n_other_contacts")})
    save(representative, False)


if __name__ == "__main__":
    main()
