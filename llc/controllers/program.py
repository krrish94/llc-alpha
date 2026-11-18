"""The torque-tracking family (protocol v4, ADR 0011): a torque program on top of a base law, run on a
hold reference, so that the realised joint torque can be compared with the command.

Programs (``case.meta["program"]``): ``hold`` (the base's frame alone: quantisation and delay noise),
``step`` (steps of ``amplitude`` times the stall torque on ``joint``, 0.5 s on, 0.5 s off), ``chirp`` (a
logarithmic chirp from ``f0`` to ``f1`` Hz at ``amplitude`` on ``joint``), ``torque_chirp`` (the chirp as a
pure torque command: kp = kd = 0, the stiffness of the frame is then the servo's own). The base is the
feedforward PD; the program adds to its feedforward term.
"""
from __future__ import annotations

import numpy as np

from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS
from llc.sim.servo import make_frame

PROGRAMS = ("hold", "step", "chirp", "torque_chirp")


def program_torque(program: dict, t: float) -> np.ndarray:
    """Additive torque of a program at time t (Nm, per joint)."""
    tau = np.zeros(NUM_JOINTS)
    kind, j, a = program["kind"], int(program.get("joint", 0)), float(program.get("amplitude", 0.0)) * TORQUE_LIMITS[int(program.get("joint", 0))]
    t0 = float(program.get("t_start", 0.5))
    if kind == "step":
        if t >= t0:
            tau[j] = a * (1.0 if int((t - t0) // 0.5) % 2 == 0 else -1.0)
    elif kind in ("chirp", "torque_chirp"):
        f0, f1, dur = float(program.get("f0", 0.5)), float(program.get("f1", 125.0)), float(program.get("duration", 2.0))
        if t0 <= t < t0 + dur:
            s = (t - t0) / dur
            phase = 2.0 * np.pi * f0 * dur * (np.power(f1 / f0, s) - 1.0) / np.log(f1 / f0)
            tau[j] = a * np.sin(phase)
    return tau


def chirp_frequency(program: dict, t: np.ndarray) -> np.ndarray:
    """Instantaneous frequency (Hz) of a chirp program at times t (NaN outside the chirp)."""
    f0, f1, dur, t0 = float(program.get("f0", 0.5)), float(program.get("f1", 125.0)), float(program.get("duration", 2.0)), float(program.get("t_start", 0.5))
    s = (np.asarray(t) - t0) / dur
    f = f0 * np.power(f1 / f0, s)
    return np.where((s >= 0) & (s < 1), f, np.nan)


class TorqueProgram:
    name = "torque_program"

    def __init__(self, base, program: dict, nominal):
        self.base, self.program, self.nominal = base, dict(program), nominal
        self.pure = self.program["kind"] == "torque_chirp"

    def reset(self) -> None:
        self.base.reset()
        self.t_frame = 0.0

    def base_term(self, q, qd_ref, qdd_ref):
        return self.base.base_term(q, qd_ref, qdd_ref)

    def command_extra(self, t: float) -> np.ndarray:
        """The program's change since the last frame was sent: the 1 kHz command the servo loop's metric compares with carries the
        program at the current time, while the frame carried it at the frame instant."""
        return program_torque(self.program, t) - program_torque(self.program, self.t_frame)

    def frame(self, t, q, qd, q_ref, qd_ref, qdd_ref, **kw) -> dict:
        self.t_frame = t
        fr = self.base.frame(t, q, qd, q_ref, qd_ref, qdd_ref)
        add = program_torque(self.program, t)
        if self.pure:  # a pure torque command: the base's feedback folded into the feedforward at this instant, no gains
            tau = fr["kp"] * (fr["q_des"] - q) + fr["kd"] * (fr["qd_des"] - qd) + fr["tau_ff"] + add
            return make_frame(q_ref, qd_ref, np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS), tau)
        fr["tau_ff"] = fr["tau_ff"] + add
        return fr

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref):  # at 1 kHz without a profile (P0): the frame's law at the state
        from llc.sim.servo import frame_law

        return frame_law(self.frame(t, q, qd, q_ref, qd_ref, qdd_ref), q, qd)
