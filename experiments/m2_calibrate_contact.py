"""Calibrate the penalty contact of the torch model against MuJoCo on transition cases: for a few
(k_n, c_n) settings, roll the same PD controller through both and compare peak force, steady
pressing force and tool penetration. Writes runs/m2/contact_calibration.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from llc.controllers.pd import PDGravity, gains_from_params, home_inertia_diag
from llc.dynamics.contact import ContactParams
from llc.dynamics.rigid_body import RigidBodyModel
from llc.eval.cases import make_case
from llc.eval.rollout import rollout
from llc.sim.nominal import NominalModel
from llc.sim.panda import PandaSim, TORQUE_LIMITS, load_model


def torch_rollout(model, case, kp, kd, params, dt=0.0025):
    ref = case.reference
    t = np.arange(0.0, ref.duration, dt)
    q_ref = torch.tensor(np.stack([np.interp(t, ref.t, ref.q[:, j]) for j in range(7)], 1), dtype=torch.float32)[None]
    qd_ref = torch.tensor(np.stack([np.interp(t, ref.t, ref.qd[:, j]) for j in range(7)], 1), dtype=torch.float32)[None]
    q = torch.tensor(case.q0, dtype=torch.float32)[None]
    qd = torch.zeros_like(q)
    kp_t, kd_t = torch.tensor(kp, dtype=torch.float32), torch.tensor(kd, dtype=torch.float32)
    lim = torch.tensor(TORQUE_LIMITS, dtype=torch.float32)
    fs, zs = [], []
    with torch.no_grad():
        for k in range(len(t)):
            g = model.bias(q, torch.zeros_like(q))
            tau = kp_t * (q_ref[:, k] - q) + kd_t * (qd_ref[:, k] - qd) + g
            tau = torch.maximum(torch.minimum(tau, lim), -lim)
            q, qd, f, p = model.step_contact(q, qd, tau, dt, params)
            fs.append(float(torch.linalg.norm(f)))
            zs.append(p[0].numpy().copy())
    return t, np.array(fs), np.array(zs)


def main() -> None:
    nominal = NominalModel()
    sim = PandaSim()
    model = RigidBodyModel(load_model(), dtype=torch.float32)
    kp, kd = gains_from_params(np.concatenate([np.log([400, 400, 300, 300, 100, 60, 30]), [0.0]]), home_inertia_diag(nominal))
    ctrl = PDGravity(kp, kd, nominal)
    cases = [make_case("transition", "tune", i, nominal) for i in range(4)]
    mj = []
    for case in cases:
        ep = rollout(sim, ctrl, case)
        press = case.reference.phase_mask("press")
        mj.append({"peak": float(ep.f_normal.max()), "steady": float(ep.f_normal[press][-200:].mean()),
                   "surface": case.meta["surface"], "speed": case.meta["approach_speed"]})
    results = []
    for k_n, c_n in [(20000, 200), (40000, 400), (60000, 600), (80000, 800), (60000, 300), (60000, 1200)]:
        params = ContactParams(k_n=k_n, c_n=c_n, mu=0.5)
        rows = []
        for case, ref_m in zip(cases, mj):
            t, f, p = torch_rollout(model, case, kp, kd, params)
            press = case.reference.phase_mask("press")
            n = len(t)
            press_t = np.array([press[min(len(press) - 1, int(round(tt / 1e-3)))] for tt in t])
            rows.append({"peak": float(f.max()), "steady": float(f[press_t][-80:].mean())})
        err = float(np.mean([abs(np.log(r["peak"] / m["peak"])) + abs(np.log(max(r["steady"], 1e-3) / m["steady"])) for r, m in zip(rows, mj)]))
        results.append({"k_n": k_n, "c_n": c_n, "torch": rows, "log_error": err})
        print(f"k_n={k_n:6d} c_n={c_n:5d}  " + " | ".join(f"peak {r['peak']:5.0f}/{m['peak']:5.0f} steady {r['steady']:4.1f}/{m['steady']:4.1f}" for r, m in zip(rows, mj)) + f"  err={err:.3f}")
    Path("runs/m2").mkdir(parents=True, exist_ok=True)
    json.dump({"mujoco": mj, "candidates": results, "gains": {"kp": kp.tolist(), "kd": kd.tolist()}}, open("runs/m2/contact_calibration.json", "w"), indent=1)


if __name__ == "__main__":
    main()
