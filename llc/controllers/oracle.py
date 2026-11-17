"""Privileged contact-impedance oracle (ADR 0008): the wanted response to a pressing or blocked
position target, implemented with information a proprioceptive controller does not have (where
the surface is, its normal, and the reference's tool path). It is the ceiling of the protocol's
contact metrics and the teacher of the distilled controllers.

Free space: the base controller's torque, unchanged. While the reference is inside a surface and
the tool has reached it, the base controller's tool-space force along the surface normal is
replaced by a bounded spring on the actual tool error,

    push = clip(f_min + k_n e_in - d_n v_in, f_min, f_cap),

with e_in the distance of the tool behind the reference along the inward normal and v_in its
velocity into the surface (the press target of the training reward, press_min + press_stiffness x
depth, on the actual error instead of the commanded depth); d_n is the damping of ratio ``zeta``
for the task-space inertia along the normal. On a hard surface the bound is reached and the force stays at ``f_cap``; on a soft one
the tool sinks and the force stays on the linear part. In blocked references the tangential force
is also replaced by a spring on the first-contact point (stop and wait); in press-and-slide
references the base controller's tangential force is kept, because the slide is the task. When
the reference leaves the surface the torque blends back to the base controller over ``blend`` s.

The bound is on the true contact force. An unmodelled payload presses on a horizontal surface with
its weight whatever the controller commands, so the oracle, being privileged, knows the payload
mass and commands ``push`` minus the weight's component along the inward normal (on a table with a
2 kg payload it lifts: the weight alone exceeds the 15 N cap). A proprioceptive student has to
infer the payload from the approach to reproduce this.
"""
from __future__ import annotations

import numpy as np

from llc.learning.data import surface_depth
from llc.sim.nominal import NominalModel

ORACLE_DEFAULTS = {"k_n": 500.0, "f_min": 5.0, "f_cap": 15.0, "zeta": 0.7, "k_t": 1000.0, "blend": 0.2}


class ContactImpedanceOracle:
    name = "oracle"

    def __init__(self, base, nominal: NominalModel, tool_ref: np.ndarray, normal, surface: str, hold_tangent: bool,
                 k_n: float = 500.0, f_min: float = 5.0, f_cap: float = 15.0, zeta: float = 0.7, k_t: float = 1000.0,
                 blend: float = 0.2, contact_eps: float = 1e-3, dt: float = 1e-3, payload_mass: float = 0.0):
        self.base, self.nominal = base, nominal
        self.weight_in = float(payload_mass * 9.81 * np.asarray(normal, float)[2])  # the payload weight's push into the surface (N)
        self.normal = np.asarray(normal, float)
        self.surface, self.hold_tangent = surface, hold_tangent
        self.tool_ref = np.asarray(tool_ref, float)
        self.depth_ref = surface_depth(self.tool_ref, surface)
        self.k_n, self.f_min, self.f_cap, self.zeta, self.k_t = k_n, f_min, f_cap, zeta, k_t
        self.blend_steps = max(1, int(round(blend / dt)))
        self.contact_eps = contact_eps
        self.reset()

    def reset(self) -> None:
        self.base.reset()
        self.k = 0
        self.anchor: np.ndarray | None = None  # tool position at the first contact while the reference is inside
        self.release_k: int | None = None  # step at which the reference left the surface (blend back to the base)
        self.push = 0.0  # last commanded pushing force (N), for analysis

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref):
        tau_base = self.base(t, q, qd, q_ref, qd_ref, qdd_ref)
        k = min(self.k, len(self.depth_ref) - 1)
        self.k += 1
        inside = bool(self.depth_ref[k] > 0.0)
        if not inside and self.anchor is None:
            self.push = 0.0
            return tau_base
        x = self.nominal.tool_pos(q)
        if inside and self.anchor is None:
            if surface_depth(x[None], self.surface)[0] <= -self.contact_eps:
                self.push = 0.0
                return tau_base  # the tool has not reached the surface yet: the approach is the base's
            self.anchor = x.copy()
        if inside:
            self.release_k = None
        elif self.release_k is None:
            self.release_k = k
        n = self.normal
        J, _ = self.nominal.tool_jacobian(q)
        v = J @ qd
        model_term = self.base.base_term(q, qd_ref, qdd_ref) if hasattr(self.base, "base_term") else self.nominal.gravity(q)
        f = np.linalg.lstsq(J.T, tau_base - model_term, rcond=None)[0]  # tool force of the base's feedback (N), its model term removed
        lam = np.linalg.inv(J @ np.linalg.inv(self.nominal.mass_matrix(q)) @ J.T + 1e-9 * np.eye(3))  # task-space inertia
        e_in = float(n @ (x - self.tool_ref[k]))  # tool behind the reference along the inward normal (m)
        v_in = float(-(n @ v))
        d_n = 2.0 * self.zeta * np.sqrt(self.k_n * max(1e-6, float(n @ lam @ n)))
        push = float(np.clip(self.f_min + self.k_n * e_in - d_n * v_in, self.f_min if inside else 0.0, self.f_cap))  # wanted true contact force
        self.push = push
        push_cmd = push - self.weight_in  # what the robot must add to the payload's weight
        f_new = f - n * (n @ f) - n * push_cmd
        if self.hold_tangent:
            P = np.eye(3) - np.outer(n, n)
            m_t = max(1e-6, float(np.trace(P @ lam @ P)) / 2.0)
            d_t = 2.0 * self.zeta * np.sqrt(self.k_t * m_t)
            f_new = -n * push_cmd + self.k_t * (P @ (self.anchor - x)) - d_t * (P @ v)
        tau = tau_base + J.T @ (f_new - f)
        if self.release_k is not None:
            a = min(1.0, (k - self.release_k) / self.blend_steps)
            tau = (1.0 - a) * tau + a * tau_base
            if a >= 1.0:
                self.anchor, self.release_k = None, None
        return tau


def oracle_for_case(base, nominal: NominalModel, case, dt: float = 1e-3, **params):
    """The oracle around ``base`` for a case; cases without a surface get the base itself."""
    normal, surface, tool_ref = case.meta.get("normal"), case.meta.get("surface"), case.meta.get("tool_ref")
    if normal is None or surface is None or tool_ref is None:
        return base
    return ContactImpedanceOracle(base, nominal, tool_ref, normal, surface, hold_tangent=(case.regime == "blocked"), dt=dt, payload_mass=float(case.perturbation.payload_mass), **{**ORACLE_DEFAULTS, **params})
