"""Milestone 4: collect diverse closed-loop trajectories from the plant (MuJoCo) for learning a
dynamics model.

    uv run python experiments/m4_collect.py --out runs/m4/data_train.npz --split train --n 24

Excitation: PD with shared gains tracking references of every regime, plus exploration torque
noise (filtered Gaussian, per-episode amplitude) and random gain scaling, so that the data cover
a range of torques, speeds and contact forces. Each episode stores q, qd, applied torque, tool
position and normal contact force at 1 kHz with its regime and seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from llc.controllers.pd import PDGravity
from llc.eval.cases import REGIMES, make_case
from llc.eval.rollout import rollout
from llc.sim.nominal import NominalModel
from llc.sim.panda import PandaSim, TORQUE_LIMITS


class NoisyPD(PDGravity):
    """PD + gravity with exploration torque noise (first-order filtered Gaussian)."""

    def __init__(self, kp, kd, nominal, rng, amplitude, tau_c=0.05, dt=1e-3):
        super().__init__(kp, kd, nominal)
        self.rng, self.amp, self.alpha = rng, amplitude, dt / tau_c
        self.noise = np.zeros(7)

    def reset(self):
        self.noise = np.zeros(7)

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref):
        self.noise += self.alpha * (-self.noise + self.rng.normal(size=7) / np.sqrt(self.alpha))
        return super().__call__(t, q, qd, q_ref, qd_ref, qdd_ref) + self.amp * TORQUE_LIMITS * self.noise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=24, help="episodes per regime")
    ap.add_argument("--regimes", nargs="+", default=list(REGIMES))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gains = json.load(open("runs/m1/gains_shared.json"))["spec"]
    nominal, sim = NominalModel(), PandaSim()
    rng = np.random.default_rng(args.seed)
    episodes = []
    for regime in args.regimes:
        for i in range(args.n):
            case = make_case(regime, args.split, i, nominal, speed_scale=float(rng.uniform(0.7, 1.4)))
            scale = float(rng.uniform(0.3, 1.5))
            amp = float(rng.choice([0.0, 0.02, 0.05, 0.1]))
            ctrl = NoisyPD(np.array(gains["kp"]) * scale, np.array(gains["kd"]) * np.sqrt(scale), nominal, rng, amp)
            ep = rollout(sim, ctrl, case)
            if ep.diverged:
                continue
            episodes.append({"regime": regime, "index": i, "gain_scale": scale, "noise_amp": amp, "q": ep.q.astype(np.float32), "qd": ep.qd.astype(np.float32),
                             "tau": ep.tau.astype(np.float32), "tool_pos": ep.tool_pos.astype(np.float32), "f_normal": ep.f_normal.astype(np.float32)})
            print(f"{case.id} gain x{scale:.2f} noise {amp:.2f}: {len(ep.t)} steps, max |qd| {np.abs(ep.qd).max():.2f}, max f {ep.f_normal.max():.0f} N", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, episodes=np.array(episodes, dtype=object), split=args.split)
    print(f"saved {len(episodes)} episodes to {args.out}")


if __name__ == "__main__":
    main()
