"""Milestone 9 (ADR 0009): is light sliding contact fragile because of the plant? Constant-force presses
(the oracle law with f_min = f_cap = F around the shared PD) on contact-family table cases, sweeping
the pressing force, the slide speed (through the case's speed scale) and the payload, on the selected
plant (LLC_PLANT): contact losses per second of slide, contact fraction and mean force.

    LLC_PLANT=v3 uv run python experiments/m9_detachment.py --gains runs/m9/gains_shared.json --out runs/m9/detachment_v3.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from llc.eval.cases_v2 import make_case_v2
from llc.eval.parallel import CaseSpec, evaluate
from llc.sim.nominal import NominalModel
from llc.sim.plant import PLANT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", default="runs/d1/gains_shared.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--forces", default="5,10,15,20,30")
    ap.add_argument("--speeds", default="0.5,1.0,2.0", help="speed scales of the contact family (slide speed 0.05 to 0.2 m/s at 1.0)")
    ap.add_argument("--payloads", default="0,2")
    ap.add_argument("--n", type=int, default=4, help="table cases per cell")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    pd = json.load(open(args.gains))["spec"]
    nominal = NominalModel()
    # table cases of the contact family (test split); the wall cases are left out so that the sweep is one geometry
    table = []
    i = 0
    while len(table) < args.n and i < 40:
        if make_case_v2("contact2", "test", i, nominal).meta["surface"] == "table":
            table.append(i)
        i += 1
    specs = {f"F{f:g}": {"kind": "oracle", "base": pd, "params": {"f_min": f, "f_cap": f}} for f in (float(x) for x in args.forces.split(","))}
    specs["pd"] = pd
    out = {"plant": PLANT.name, "gains": args.gains, "cases": table, "cells": []}
    for s in (float(x) for x in args.speeds.split(",")):
        for pl in (float(x) for x in args.payloads.split(",")):
            cases = [CaseSpec("contact2", "test", k, protocol="v2", speed_scale=s, payload=pl) for k in table]
            res = evaluate(specs, cases, workers=args.workers)
            for name, rows in res.items():
                cell = {"controller": name, "speed_scale": s, "payload": pl,
                        "contact_loss_rate": float(np.nanmean([m["contact_loss_rate"] for m in rows])), "contact_fraction": float(np.nanmean([m["contact_fraction"] for m in rows])),
                        "f_mean": float(np.nanmean([m["f_mean"] for m in rows])), "f_peak_20ms": float(np.nanmean([m["f_peak_20ms"] for m in rows])), "other_contact": float(np.nanmean([m["other_contact"] for m in rows]))}
                out["cells"].append(cell)
                print(f"{PLANT.name} speed x{s:g} payload {pl:g} kg {name:5s}: losses/s {cell['contact_loss_rate']:6.1f} contact {cell['contact_fraction']:.2f} f_mean {cell['f_mean']:5.1f} other {cell['other_contact']:.2f}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
