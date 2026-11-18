"""Conventional joint-space controllers, running at the physics rate (1 kHz).

All controllers share the call signature
    tau = controller(t, q, qd, q_ref, qd_ref, qdd_ref)
and a ``reset()`` method. Model-based terms use the nominal model only.
"""
from __future__ import annotations

import numpy as np

from llc.sim.nominal import NominalModel
from llc.sim.panda import HOME, NUM_JOINTS


def gains_from_params(params: np.ndarray, inertia_diag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map the tuning parameters (log stiffness per joint, log damping ratio) to (kp, kd).

    kd_i = 2 zeta sqrt(kp_i * M_ii) makes every joint critically damped at zeta = 1 with respect to
    the nominal diagonal inertia ``inertia_diag`` (taken at the home configuration).
    """
    kp = np.exp(params[:NUM_JOINTS])
    zeta = np.exp(params[NUM_JOINTS])
    kd = 2.0 * zeta * np.sqrt(kp * inertia_diag)
    return kp, kd


def home_inertia_diag(nominal: NominalModel) -> np.ndarray:
    return np.diag(nominal.mass_matrix(HOME)).copy()


class PDGravity:
    """tau = kp (q_ref - q) + kd (qd_ref - qd) + g(q), with g from the nominal model."""

    name = "pd_gravity"

    def __init__(self, kp: np.ndarray, kd: np.ndarray, nominal: NominalModel):
        self.kp, self.kd, self.nominal = np.asarray(kp, float), np.asarray(kd, float), nominal

    def reset(self) -> None:
        pass

    def base_term(self, q, qd_ref, qdd_ref):
        return self.nominal.gravity(q)

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref):
        return self.kp * (q_ref - q) + self.kd * (qd_ref - qd) + self.nominal.gravity(q)

    def frame(self, t, q, qd, q_ref, qd_ref, qdd_ref) -> dict:
        """The impedance frame of this law at a host instant (ADR 0011): its gains, the reference and its model term."""
        from llc.sim.servo import make_frame

        return make_frame(q_ref, qd_ref, self.kp, self.kd, self.base_term(q, qd_ref, qdd_ref))


class PDFeedforward(PDGravity):
    """PD + gravity plus the nominal model's inverse-dynamics feedforward of the reference,
    tau = kp e + kd de + M(q) qdd_ref + C(q, qd_ref) qd_ref + g(q): the classical anticipatory
    baseline against which a controller that previews the target must be compared (ADR 0008)."""

    name = "pd_feedforward"

    def base_term(self, q, qd_ref, qdd_ref):
        return self.nominal.mass_matrix(q) @ qdd_ref + self.nominal.bias(q, qd_ref)

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref):
        return self.kp * (q_ref - q) + self.kd * (qd_ref - qd) + self.base_term(q, qd_ref, qdd_ref)


class ComputedTorque:
    """Inverse-dynamics (computed-torque) control with the nominal model, a privileged baseline:
    tau = M(q)(qdd_ref + kp e + kd de) + C(q,qd) qd + g(q). Here kp, kd are in acceleration units
    (kp = w^2, kd = 2 zeta w) and the reference acceleration must exist."""

    name = "computed_torque"

    def __init__(self, omega: np.ndarray, zeta: float, nominal: NominalModel):
        self.kp = np.asarray(omega, float) ** 2
        self.kd = 2.0 * zeta * np.asarray(omega, float)
        self.nominal = nominal

    def reset(self) -> None:
        pass

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref):
        acc = qdd_ref + self.kp * (q_ref - q) + self.kd * (qd_ref - qd)
        return self.nominal.mass_matrix(q) @ acc + self.nominal.bias(q, qd)


class GainScheduledPD(PDGravity):
    """PD + gravity whose gains follow a schedule of (time, kp, kd) breakpoints with linear
    interpolation over ``blend`` seconds after each switch (Milestone 6 baseline)."""

    name = "pd_scheduled"

    def __init__(self, schedule: list[tuple[float, np.ndarray, np.ndarray]], blend: float, nominal: NominalModel):
        super().__init__(schedule[0][1], schedule[0][2], nominal)
        self.schedule, self.blend = schedule, blend

    def gains_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        kp, kd = self.schedule[0][1], self.schedule[0][2]
        for t_switch, kp_new, kd_new in self.schedule[1:]:
            if t >= t_switch:
                a = 1.0 if self.blend <= 0 else min(1.0, (t - t_switch) / self.blend)
                kp, kd = (1 - a) * kp + a * kp_new, (1 - a) * kd + a * kd_new
        return kp, kd

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref):
        kp, kd = self.gains_at(t)
        return kp * (q_ref - q) + kd * (qd_ref - qd) + self.nominal.gravity(q)
