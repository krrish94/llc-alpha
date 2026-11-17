"""Build controllers from plain, picklable specifications (used by parallel evaluation)."""
from __future__ import annotations

import numpy as np

from llc.controllers.pd import ComputedTorque, GainScheduledPD, PDFeedforward, PDGravity
from llc.sim.nominal import NominalModel


def build_controller(spec: dict, nominal: NominalModel, case=None):
    kind = spec["kind"]
    if kind == "pd_phase":
        # gains per phase label (Milestone 6); the schedule follows the reference's phase boundaries
        phases = case.reference.phases
        schedule = []
        for t0, _, label in phases:
            g = spec["phase_gains"].get(label, spec["phase_gains"]["default"])
            schedule.append((float(t0), np.asarray(g["kp"]), np.asarray(g["kd"])))
        return GainScheduledPD(schedule, float(spec["blend"]), nominal)
    if kind == "pd_gravity":
        return PDGravity(np.asarray(spec["kp"]), np.asarray(spec["kd"]), nominal)
    if kind == "pd_feedforward":
        return PDFeedforward(np.asarray(spec["kp"]), np.asarray(spec["kd"]), nominal)
    if kind == "oracle":  # privileged contact-impedance oracle around a base controller (ADR 0008); needs the case
        from llc.controllers.oracle import oracle_for_case

        return oracle_for_case(build_controller(spec["base"], nominal, case), nominal, case, **spec.get("params", {}))
    if kind == "computed_torque":
        return ComputedTorque(np.asarray(spec["omega"]), float(spec["zeta"]), nominal)
    if kind == "pd_scheduled":
        schedule = [(float(t), np.asarray(kp), np.asarray(kd)) for t, kp, kd in spec["schedule"]]
        return GainScheduledPD(schedule, float(spec["blend"]), nominal)
    if kind == "learned":
        from llc.controllers.learned import load_learned_controller

        return load_learned_controller(spec["path"], nominal, **spec.get("options", {}))
    raise ValueError(f"unknown controller kind {kind}")
