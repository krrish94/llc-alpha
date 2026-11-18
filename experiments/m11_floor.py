"""The error floor of the model-based interface (Milestone 11, Phase 2a diagnostic): on fixed hold windows with torque
programs, the realised-torque loss of the model-based interface per profile, as is and with the feedback quantisation,
the host period or the bus delay removed one at a time. Says what a learned correction on the profile's channels can
and cannot remove (the servo acts on its own quantised measurement between host frames).

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/m11_floor.py --out runs/m11/floor.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from experiments.m11_interface import rollout_interface, sample_programs
from llc.dynamics.rigid_body import RigidBodyModel
from llc.learning.data import reference_pool_v2, sample_batch_v2
from llc.learning.interface import LIMITS_T, InterfaceConfig, InterfacePolicy
from llc.learning.mjgrad import MjBatch
from llc.sim.instances import sample_instance
from llc.sim.nominal import NominalModel
from llc.sim.panda import load_model

VARIANTS = {"as_is": {}, "exact_velocity": {"vel_res": 0.0}, "exact_feedback": {"vel_res": 0.0, "pos_res": 0.0, "goal_res": 0.0}, "no_delay": {"delay": 0.0},
            "period_1ms": {"period": 1e-3}, "period_1ms_no_delay": {"period": 1e-3, "delay": 0.0}, "all_removed": {"period": 1e-3, "delay": 0.0, "vel_res": 0.0, "pos_res": 0.0, "goal_res": 0.0}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", default="runs/m9/gains_shared.json")
    ap.add_argument("--profiles", nargs="*", default=["p1", "p2", "p3"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mix", default="hold", help="hold: hold windows with torque programs; free: free-space windows under the feedforward PD")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(3)
    rng = np.random.default_rng(5)
    gains = json.load(open(args.gains))["spec"]
    model = RigidBodyModel(load_model(), dtype=torch.float32)
    pool = reference_pool_v2(48, nominal=NominalModel())
    weights = {"hold": 1.0} if args.mix == "hold" else {"free": 0.5, "policy": 0.25, "step": 0.25}
    q_ref, qd_ref, q0, qd0, labels, f_ext, _ = sample_batch_v2(pool, weights, args.batch, 1000, 1e-3, rng, pulse_prob=0.3)
    programs = sample_programs(labels, rng)
    batch = MjBatch([sample_instance(rng, payload_mass=float(rng.uniform(0, 3))) for _ in range(args.batch)], threads=4)
    out = {"mix": args.mix, "batch": args.batch, "programs": [p["kind"] if p else None for p in programs], "profiles": {}}
    for prof in args.profiles:
        out["profiles"][prof] = {}
        for name, ov in VARIANTS.items():
            pol = InterfacePolicy(InterfaceConfig(prof, tuple(gains["kp"]), tuple(gains["kd"]), 10, 32, 1, overrides=ov or None))
            with torch.no_grad():
                r = rollout_interface(pol, batch, model, torch.as_tensor(q_ref), torch.as_tensor(qd_ref), q0, qd0, programs, f_ext, train=False, obs_noise_q=0.0)
            rmse = float(torch.sqrt((((r["tau"] - r["tau_cmd"]) / LIMITS_T) ** 2).mean()))
            out["profiles"][prof][name] = {"track_loss": r["terms"]["track"], "rate_loss": r["terms"]["rate"], "tau_track_rmse": rmse}
            print(f"{prof} {name:22s} track {r['terms']['track']:.5f} (rmse {rmse:.4f} of the limits) rate {r['terms']['rate']:.4f}", flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
