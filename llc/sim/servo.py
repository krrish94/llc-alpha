"""Interface profiles and the servo loop (ADR 0011): what a host can command a position-controlled
arm, how often, with what resolution, and what the servo holds between host frames.

The upper controller is called once per host frame with the quantised feedback and returns an
impedance frame (q_des, qd_des, kp, kd, tau_ff, cap), the general form of every command the project's
controllers produce (a pure torque command is the frame with kp = kd = 0). The model-based interface
maps the frame to the profile's channels (decision 3 of ADR 0011): channels the profile lacks are
folded into the goal position so that the servo's torque equals the command at the frame's arrival,
predicted over the bus delay (the reference advanced by its velocity to the middle of the hold, the measured state by its own to the arrival). The frame is quantised, queued by the bus delay and held; the servo law
runs at the physics rate on the quantised measurement,

    tau = kp_s (q_goal - q_m) + kd_s (qd_goal - qd_m) + tau_ff_s,   |tau| <= cap.

On the Panda (the interface ablation) the servo's position loop is this torque-level PD with the
Panda's torque limits as the cap; the real arms add the DC-motor layer underneath (ADR 0011).

``ServoLoop`` has the controller call signature and returns the realised torque at every physics
step; ``tau_command`` holds the latest upper frame's law with the current reference evaluated at the true
state at that step (what a 1 kHz torque interface would apply), the command the torque-tracking metrics
compare the realised torque with: the hold between frames and the bus delay both count as error.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS

FRAME_KEYS = ("q_des", "qd_des", "kp", "kd", "tau_ff", "cap")


@dataclass(frozen=True)
class ServoProfile:
    name: str
    period: float  # s, host frame period
    delay: float  # s, bus delay from the frame's send to its effect
    pos_res: float  # rad, feedback position resolution (0 = exact)
    vel_res: float  # rad/s, feedback velocity resolution
    goal_res: float  # rad, goal position resolution
    gains: str  # "frame": gains in every frame (MIT mode); "ram": written per frame; "fixed": the servo's own
    kp_range: tuple[float, float]  # multiples of the servo's nominal stiffness the host may command
    kd_range: tuple[float, float]  # multiples of the nominal damping
    gain_bits: int  # quantisation of a gain channel over its range (0 = exact)
    vel_cmd: bool  # velocity setpoint channel
    vel_cmd_res: float  # rad/s
    tau_ff: bool  # feedforward torque channel
    tau_bits: int  # quantisation of the torque channel over +-limit
    cap_cmd: bool  # per-frame torque (current) cap channel
    torque_res: float = 0.0  # Nm per count of the torque feedback the controller sees (0 = exact; the real arms' current resolution)
    vel_filter: float = 0.0  # s, time constant of a first-order filter on the quantised velocity feedback (0 = raw); the servo damps on, and reports, the filtered estimate (ADR 0011 amendment 3)

    @property
    def period_steps(self) -> int:
        return max(1, int(round(self.period / 1e-3)))

    @property
    def delay_steps(self) -> int:
        return int(round(self.delay / 1e-3))


# Profiles of ADR 0011, decision 1. P0 reproduces the torque interface of reports 01 to 18 exactly.
PROFILES = {
    "p0": ServoProfile("p0", 1e-3, 0.0, 0.0, 0.0, 0.0, "frame", (0.0, 1e9), (0.0, 1e9), 0, True, 0.0, True, 0, False),
    # MIT mode on CAN (DM4310/DM4340): 16-bit position over 25 rad, 12-bit velocity over 60 rad/s, 12-bit Kp over 0..500 Nm/rad
    # (6.25 times the YAM's shipped 80), 12-bit Kd over 0..5 (the shipped damping is the range's end), 12-bit feedforward torque.
    "p1": ServoProfile("p1", 4e-3, 1e-3, 25.0 / 65536, 60.0 / 4096, 25.0 / 65536, "frame", (0.0, 6.25), (0.0, 1.0), 12, True, 60.0 / 4096, True, 12, False),
    # Dynamixel X position mode: 4096 counts per revolution, velocity 0.229 rpm, RAM P and D gains (no practical upper limit;
    # a lower bound keeps the goal offset finite), a current cap per frame, no velocity or torque channel.
    "p2": ServoProfile("p2", 10e-3, 2e-3, 2 * np.pi / 4096, 0.229 * 2 * np.pi / 60, 2 * np.pi / 4096, "ram", (0.05, 6.0), (0.05, 6.0), 12, False, 0.0, False, 0, True, 2.69e-3 * 1.78),
    # Feetech STS position mode: 4096 counts, velocity 50 steps/s, gains in EEPROM (fixed), a torque limit per frame.
    "p3": ServoProfile("p3", 20e-3, 3e-3, 2 * np.pi / 4096, 50 * 2 * np.pi / 4096, 2 * np.pi / 4096, "fixed", (1.0, 1.0), (1.0, 1.0), 0, False, 0.0, False, 0, True, 6.5e-3 * 1.09),
}


def get_profile(name: str, **overrides) -> ServoProfile:
    return replace(PROFILES[name], **overrides) if overrides else PROFILES[name]


def quantise(x: np.ndarray, res: float) -> np.ndarray:
    return x if res <= 0.0 else np.round(np.asarray(x) / res) * res


def quantise_range(x: np.ndarray, lo, hi, bits: int) -> np.ndarray:
    """Clip to [lo, hi] and quantise to ``bits`` over the range (per joint; 0 bits = clip only)."""
    x = np.clip(x, lo, hi)
    if bits <= 0:
        return x
    n = 2**bits - 1
    return lo + np.round((x - lo) / (hi - lo + 1e-12) * n) / n * (hi - lo)


def frame_law(frame: dict, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """The frame's own law at a state: kp (q_des - q) + kd (qd_des - qd) + tau_ff, capped."""
    tau = frame["kp"] * (frame["q_des"] - q) + frame["kd"] * (frame["qd_des"] - qd) + frame["tau_ff"]
    return np.clip(tau, -frame["cap"], frame["cap"])


def make_frame(q_des, qd_des, kp, kd, tau_ff, cap=None) -> dict:
    return {"q_des": np.asarray(q_des, float), "qd_des": np.asarray(qd_des, float), "kp": np.asarray(kp, float), "kd": np.asarray(kd, float),
            "tau_ff": np.asarray(tau_ff, float), "cap": TORQUE_LIMITS.copy() if cap is None else np.asarray(cap, float)}


class ModelInterface:
    """The model-based interface (ADR 0011, decision 3): maps an upper frame to the profile's channels
    so that the servo's torque equals the frame's law at the frame's predicted arrival state.
    ``kp_nominal``, ``kd_nominal`` are the servo's own gains (the shipped driver's; the tuned PD gains
    on the Panda), used as the fixed gains of P3, as the stiffness a torque-only command (kp = 0) is
    realised with, and as the unit of the gain ranges."""

    def __init__(self, profile: ServoProfile, kp_nominal, kd_nominal):
        self.p = profile
        self.kp_n, self.kd_n = np.asarray(kp_nominal, float), np.asarray(kd_nominal, float)
        self.limits = TORQUE_LIMITS

    def reset(self) -> None:
        pass

    def servo_gains(self, frame: dict) -> tuple[np.ndarray, np.ndarray]:
        p = self.p
        if p.gains == "fixed":
            return self.kp_n.copy(), self.kd_n.copy()
        kp = np.where(frame["kp"] > 0.0, frame["kp"], self.kp_n)  # a torque-only command is held with the servo's own spring
        kd = np.where(frame["kp"] > 0.0, frame["kd"], self.kd_n)
        kp = quantise_range(kp, p.kp_range[0] * self.kp_n, p.kp_range[1] * self.kp_n, p.gain_bits)
        kd = quantise_range(kd, p.kd_range[0] * self.kd_n, p.kd_range[1] * self.kd_n, p.gain_bits)
        return kp, kd

    def __call__(self, frame: dict, q_m: np.ndarray, qd_m: np.ndarray) -> dict:
        """The servo frame (quantised to the profile's channels) for an upper frame at the measured state."""
        p = self.p
        kp_s, kd_s = self.servo_gains(frame)
        cap = np.minimum(frame["cap"], self.limits) if p.cap_cmd else self.limits.copy()
        frame = {**frame, "q_des": frame["q_des"] + frame["qd_des"] * (p.delay + 0.5 * (p.period - 1e-3))}  # the setpoint for the middle of the frame's hold: the reference advanced over the bus delay and half the hold (zero-order-hold compensation; zero on P0)
        if p.tau_ff and p.vel_cmd and p.gains == "frame":  # MIT mode: the frame passes through, its torque beyond the gains' reach into tau_ff
            tau_ff = frame["tau_ff"] + (frame["kp"] - kp_s) * (frame["q_des"] - q_m) + (frame["kd"] - kd_s) * (frame["qd_des"] - qd_m)
            q_goal, qd_goal = quantise(frame["q_des"], p.goal_res), quantise(frame["qd_des"], p.vel_cmd_res)
            tau_ff = quantise_range(tau_ff, -self.limits, self.limits, p.tau_bits)
            return {"q_des": q_goal, "qd_des": qd_goal, "kp": kp_s, "kd": kd_s, "tau_ff": tau_ff, "cap": cap}
        # position interfaces: fold the command into the goal at the predicted arrival state
        q_p = q_m + qd_m * p.delay
        tau_c = frame_law(frame, q_p, qd_m)
        q_goal = quantise(q_p + (tau_c + kd_s * qd_m) / kp_s, p.goal_res)
        return {"q_des": q_goal, "qd_des": np.zeros(NUM_JOINTS), "kp": kp_s, "kd": kd_s, "tau_ff": np.zeros(NUM_JOINTS), "cap": cap}


class ServoLoop:
    """Runs an upper controller through a profile (ADR 0011, decision 2). Controller signature; returns
    the servo's torque at every physics step. The upper controller must provide ``frame(t, q, qd, q_ref,
    qd_ref, qdd_ref) -> dict`` (every controller of ``llc.controllers`` does) and is called once per
    host frame with the quantised feedback; its ``last_tau`` is kept at the realised torque so that a
    learned controller's history sees what the servo applied. ``interface`` is the model-based mapping
    or a learned one with the same call signature (``reset``, ``(frame, q_m, qd_m) -> servo frame``)."""

    name = "servo"

    def __init__(self, inner, profile: ServoProfile, interface: ModelInterface, dt: float = 1e-3):
        self.inner, self.p, self.interface, self.dt = inner, profile, interface, dt
        self.period, self.delay = max(1, int(round(profile.period / dt))), int(round(profile.delay / dt))
        self.preview = getattr(inner, "preview", 0)
        self.policy = getattr(inner, "policy", None)
        self.reset()

    def reset(self) -> None:
        self.inner.reset()
        self.interface.reset()
        self.k = 0
        self.pending: deque = deque()  # (step at which it takes effect, servo frame, upper frame)
        self.held: dict | None = None
        self.held_upper: dict | None = None
        self.latest_upper: dict | None = None  # the most recent upper frame, in flight or held: the command the metrics compare with
        self.last_tau = np.zeros(NUM_JOINTS)
        self.tau_command = np.zeros(NUM_JOINTS)  # the upper frame's law at the true state (torque-tracking metrics)
        self.n_frames = 0
        self.qd_f: np.ndarray | None = None  # the filtered velocity estimate (profile.vel_filter > 0)

    def measure(self, q, qd) -> tuple[np.ndarray, np.ndarray]:
        """The feedback the servo acts on and the host sees: quantised position and velocity, the velocity through the
        profile's first-order filter when it has one."""
        q_m, qd_m = quantise(q, self.p.pos_res), quantise(qd, self.p.vel_res)
        if self.p.vel_filter > 0.0:
            a = self.dt / (self.p.vel_filter + self.dt)
            self.qd_f = np.asarray(qd_m, float).copy() if self.qd_f is None else self.qd_f + a * (qd_m - self.qd_f)
            qd_m = self.qd_f
        return q_m, qd_m

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref, q_ref_future=None):
        q_m, qd_m = self.measure(q, qd)
        if self.k % self.period == 0:
            kw = {"q_ref_future": q_ref_future} if (self.preview and q_ref_future is not None) else {}
            upper = self.inner.frame(t, q_m, qd_m, q_ref, qd_ref, qdd_ref, **kw)
            self.pending.append((self.k + self.delay, self.interface(upper, q_m, qd_m), upper))
            self.latest_upper = upper
            self.n_frames += 1
        while self.pending and self.k >= self.pending[0][0]:
            _, self.held, self.held_upper = self.pending.popleft()
        self.k += 1
        if self.held is None:  # before the first frame arrives: gravity hold with the servo's own gains around the start
            kp, kd = self.interface.servo_gains(make_frame(q_m, 0 * qd_m, 0 * q_m, 0 * q_m, 0 * q_m))
            self.held = {"q_des": q_m.copy(), "qd_des": np.zeros(NUM_JOINTS), "kp": kp, "kd": kd, "tau_ff": self.inner.base_term(q_m, np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS)) if hasattr(self.inner, "base_term") else np.zeros(NUM_JOINTS), "cap": TORQUE_LIMITS.copy()}
            self.held_upper = self.held
        tau = frame_law(self.held, q_m, qd_m)
        # the command the torque-tracking metrics compare with: the latest upper frame's law with the current reference at the true
        # state, what a 1 kHz torque interface would apply now (the hold and the bus delay both show as error)
        latest = self.latest_upper or self.held_upper
        self.tau_command = frame_law({**latest, "q_des": np.asarray(q_ref, float), "qd_des": np.asarray(qd_ref, float)}, q, qd) + (self.inner.command_extra(t) if hasattr(self.inner, "command_extra") else 0.0)
        self.last_tau = tau
        self.inner.last_tau = tau
        self.interface.last_tau = tau
        return tau
