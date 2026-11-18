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
    if kind == "servo":  # an upper controller through an interface profile (ADR 0011)
        from llc.controllers.learned import LearnedController
        from llc.sim.servo import ModelInterface, ServoLoop, get_profile

        profile = get_profile(spec["profile"], **spec.get("overrides", {}))
        inner_spec = spec["inner"]
        if inner_spec["kind"] == "learned":  # the policy decides at every host frame, no compute latency
            inner_spec = {**inner_spec, "options": {**inner_spec.get("options", {}), "latency_s": 0.0, "dt": profile.period}}
        inner = build_controller(inner_spec, nominal, case)
        iface = spec.get("interface", "model")
        if iface == "model":
            interface = ModelInterface(profile, np.asarray(spec["kp"]), np.asarray(spec["kd"]))
        else:
            from llc.learning.interface import load_learned_interface

            interface = load_learned_interface(iface["path"], profile, np.asarray(spec["kp"]), np.asarray(spec["kd"]), nominal)
        return ServoLoop(inner, profile, interface)
    if kind == "pd_per_family":  # the per-family PD (ADR 0012): the gain set tuned for the case's family, the shared set otherwise
        gains = spec["gains"]
        regime = getattr(case, "regime", None)
        return build_controller(gains.get(regime, gains.get("shared")), nominal, case)
    if kind == "motor":  # the DC-motor layer under a controller, normally a servo loop (ADR 0011 amendment 3)
        from llc.sim.motor import MotorLayer, params_for

        return MotorLayer(build_controller(spec["inner"], nominal, case), params_for(spec.get("params", "nominal"), case.seed if case is not None else 0))
    if kind == "torque_program":  # the torque-tracking family (protocol v4): a PD frame plus a torque program from the case
        from llc.controllers.program import TorqueProgram

        return TorqueProgram(build_controller(spec["base"], nominal, case), (case.meta if case is not None else {}).get("program", {"kind": "hold"}), nominal)
    if kind == "learned":
        from llc.controllers.learned import load_learned_controller

        options = dict(spec.get("options", {}))
        if case is not None:  # ADR 0012: the episode's base gains and compute latency, unless the spec sets them
            meta = getattr(case, "meta", {}) or {}
            if "base_scale" in meta:
                options.setdefault("base_scale", meta["base_scale"])
            if "latency_s" in meta:
                options.setdefault("latency_s", meta["latency_s"])
        return load_learned_controller(spec["path"], nominal, **options)
    raise ValueError(f"unknown controller kind {kind}")
