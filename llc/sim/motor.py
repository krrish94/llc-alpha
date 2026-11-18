"""The DC-motor layer under the servo law (ADR 0011, decision 2 and amendment 3, item 5).

On a real arm the servo's torque command becomes a current command, and the realised joint torque differs
from it by the actuator's physics: the firmware's torque constant (a gain-conversion error), the current
loop's lag together with the frame timing, the voltage limit (the torque the motor can drive at a speed falls
with the back-EMF, the torque-speed envelope; braking torque is not limited this way), viscous friction, and
Coulomb friction beyond what the plant already models. The layer wraps a controller (normally a ``ServoLoop``)
and applies these to its torque at every physics step; the torque feedback the upper layers see is what the
firmware reports, the commanded current times its own constant, quantised to the current resolution.

The constants are unknown without hardware. ``nominal`` is the layer with none of the effects (the servo
abstraction of the reports), ``sample`` draws every unknown per episode from the ranges of ``RANGES`` (relative
to the plant's torque and velocity limits), which is the randomisation the qualification of decision 9 asks for
in its robustness form.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS
from llc.sim.plant import PLANT

VELOCITY_LIMITS = np.asarray(PLANT.velocity_limits, float)

# Ranges of the unknowns, relative to the plant's limits where dimensionful (ADR 0011 amendment 3).
RANGES = {
    "kt_scale": (0.85, 1.15),  # realised torque per commanded torque (the firmware's torque constant against the true one)
    "stall_per_peak": (1.5, 3.0),  # the voltage-limited stall torque as a multiple of the current-limited peak (the torque limit)
    "nl_per_vlimit": (0.8, 1.2),  # the no-load speed as a multiple of the plant's velocity limit
    "viscous_frac": (0.0, 0.05),  # viscous torque at the velocity limit as a fraction of the peak torque
    "coulomb_frac": (0.0, 0.02),  # Coulomb torque as a fraction of the peak torque, beyond the plant's own friction loss
    "lag": (0.0, 2e-3),  # s, first-order lag of the current loop plus the frame timing
    "res_per_peak": (0.0, 2.0 / 4096),  # torque-feedback resolution as a fraction of the peak torque (12 bits over the range)
}


@dataclass(frozen=True)
class MotorParams:
    kt_scale: np.ndarray  # per joint
    tau_stall: np.ndarray  # Nm
    omega_nl: np.ndarray  # rad/s
    viscous: np.ndarray  # Nm s/rad
    coulomb: np.ndarray  # Nm
    lag: float  # s
    torque_res: float  # Nm per count of the torque feedback (0 = exact)

    def as_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in self.__dict__.items()}


def nominal() -> MotorParams:
    """No gain error, the envelope above the limits at every speed up to the velocity limit, no friction, no lag."""
    return MotorParams(np.ones(NUM_JOINTS), 1e3 * TORQUE_LIMITS, 1e3 * VELOCITY_LIMITS, np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS), 0.0, 0.0)


def sample(rng: np.random.Generator) -> MotorParams:
    u = lambda key, n=None: rng.uniform(*RANGES[key], size=n)
    return MotorParams(u("kt_scale", NUM_JOINTS), u("stall_per_peak", NUM_JOINTS) * TORQUE_LIMITS, u("nl_per_vlimit", NUM_JOINTS) * VELOCITY_LIMITS,
                       u("viscous_frac", NUM_JOINTS) * TORQUE_LIMITS / VELOCITY_LIMITS, u("coulomb_frac", NUM_JOINTS) * TORQUE_LIMITS, float(u("lag")), float(u("res_per_peak")) * float(TORQUE_LIMITS.max()))


def params_for(spec, seed: int) -> MotorParams:
    """``"nominal"``, ``"random"`` (drawn from the episode's seed, the same instance for every controller on a case) or a dict."""
    if spec == "nominal":
        return nominal()
    if spec == "random":
        return sample(np.random.default_rng(int(seed) + 7919))
    return MotorParams(**{k: (np.asarray(v, float) if isinstance(v, list) else float(v)) for k, v in spec.items()})


class MotorLayer:
    """A controller wrapper: the inner controller's torque through the motor's physics at every physics step."""

    name = "motor"

    def __init__(self, inner, params: MotorParams, dt: float = 1e-3):
        self.inner, self.m, self.dt = inner, params, dt
        self.preview = getattr(inner, "preview", 0)
        self.policy = getattr(inner, "policy", None)
        self.reset()

    def reset(self) -> None:
        self.inner.reset()
        self.tau_f = np.zeros(NUM_JOINTS)  # the current loop's state
        self.last_tau = np.zeros(NUM_JOINTS)
        self.tau_command = np.zeros(NUM_JOINTS)

    def feedback(self, tau_i: np.ndarray) -> np.ndarray:
        """What the firmware reports: its current times its own torque constant, quantised."""
        fb = tau_i / self.m.kt_scale
        return fb if self.m.torque_res <= 0.0 else np.round(fb / self.m.torque_res) * self.m.torque_res

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref, q_ref_future=None):
        kw = {"q_ref_future": q_ref_future} if (self.preview and q_ref_future is not None) else {}
        tau_cmd = np.asarray(self.inner(t, q, qd, q_ref, qd_ref, qdd_ref, **kw), float)
        m = self.m
        tau_i = np.clip(m.kt_scale * tau_cmd, -TORQUE_LIMITS, TORQUE_LIMITS)  # the current the firmware commands, in true torque
        if m.lag > 0.0:
            self.tau_f += self.dt / (m.lag + self.dt) * (tau_i - self.tau_f)
            tau_i = self.tau_f.copy()
        else:
            self.tau_f = tau_i.copy()
        envelope = np.clip(m.tau_stall * (1.0 - np.abs(qd) / m.omega_nl), 0.0, TORQUE_LIMITS)  # the torque the voltage can drive at this speed
        driving = tau_i * qd > 0.0
        tau_i = np.where(driving, np.clip(tau_i, -envelope, envelope), tau_i)
        tau = tau_i - m.viscous * qd - m.coulomb * np.tanh(qd / 0.05)
        fb = self.feedback(tau_i)
        for obj in (self.inner, getattr(self.inner, "inner", None), getattr(self.inner, "interface", None)):  # the servo loop, its upper controller and interface see the firmware's torque feedback
            if obj is not None and hasattr(obj, "last_tau"):
                obj.last_tau = fb
        self.tau_command = getattr(self.inner, "tau_command", tau_cmd)
        self.last_tau = tau
        return tau
