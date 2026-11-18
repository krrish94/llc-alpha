"""Run a learned policy inside the 1 kHz simulator loop with its network rate, latency and history.

Timing (ADR 0002): the network observes at physics step k (every ``period`` steps), its output
takes effect ``latency`` steps later and is held until the next output takes effect. The fast
loop applies the held output at every physics step (numpy, same law as ``Policy.fast_torque``).
"""
from __future__ import annotations

import numpy as np
import torch

from llc.learning.policy import HistoryBuffer, OutputFilter, Policy, load_policy
from llc.sim.nominal import NominalModel
from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS


class LearnedController:
    name = "learned"

    def __init__(self, policy: Policy, nominal: NominalModel, dt: float = 1e-3, latency_s: float | None = None, base_scale=None):
        """``latency_s`` overrides the policy's compute latency (ADR 0010 realism check): a decision made
        at step k takes effect ``latency`` steps later, in both the period path and the hold path.
        ``base_scale`` = (stiffness multipliers, damping multipliers) of the base PD for this episode (ADR 0012): the
        fast loop runs at the scaled gains and the policy sees the scales when it has the input."""
        from llc.learning.policy import encode_extras

        self.policy, self.nominal = policy.eval(), nominal
        cfg = policy.cfg
        self.period = max(1, int(round(1.0 / (cfg.rate_hz * dt))))
        self.latency = int(round((cfg.latency_s if latency_s is None else latency_s) / dt))
        self.hold = max(1, int(getattr(cfg, "hold_steps", 1)))
        assert self.hold == 1 or self.period == 1, "hold_steps applies to 1 kHz policies"
        assert self.latency >= 0, "latency must be non-negative"  # decisions in flight are queued, so a latency longer than the period is allowed
        self.kind, self.history = cfg.kind, cfg.history
        self.preview = cfg.preview  # > 0: the runner passes the future targets (preview_indices) at every call
        self.feedforward = cfg.feedforward  # the PD base carries the nominal inverse-dynamics feedforward of the reference
        kp_scale, kd_scale = (np.ones(NUM_JOINTS), np.ones(NUM_JOINTS)) if base_scale is None else (np.asarray(base_scale[0], float), np.asarray(base_scale[1], float))
        self.base_scale = (kp_scale, kd_scale)
        self.kp_shared = np.asarray(cfg.kp_shared) * kp_scale
        self.kd_shared = np.asarray(cfg.kd_shared) * kd_scale
        self.extras = encode_extras(cfg, kp_scale, kd_scale, self.latency * dt) if cfg.extra_dim > 0 else None
        self.limits_t = torch.tensor(TORQUE_LIMITS, dtype=torch.float32)
        self.dt = dt
        self.inverse_dynamics = None
        if cfg.ext_torque:  # the robot's own model: M(q) qdd + bias(q, qd), torch (1,7) in and out
            def inverse_dynamics(q, qd, qdd):
                qn, qdn, qddn = (x[0].numpy().astype(np.float64) for x in (q, qd, qdd))
                return torch.as_tensor(nominal.mass_matrix(qn) @ qddn + nominal.bias(qn, qdn), dtype=torch.float32)[None]
            self.inverse_dynamics = inverse_dynamics
        self.reset()

    def reset(self) -> None:
        from collections import deque

        self.k = 0
        self.held: dict[str, np.ndarray] | None = None
        self.pending: deque = deque()  # (step at which it takes effect, output) of the decisions in flight
        self.buffer: HistoryBuffer | None = None
        self.last_tau = np.zeros(NUM_JOINTS)
        self.n_updates = 0
        self.last_obs = None  # observation of the last network update (distillation, probes)
        self.filter = OutputFilter(self.policy.cfg.output_tau, self.dt)

    @property
    def hist_q(self):  # for tests: strided-then-dense history windows as numpy
        return self.buffer.windows()[0][0].numpy() if self.buffer is not None else None

    def _network_update(self, q, qd, q_ref, qd_ref, evaluate: bool = True, q_ref_future=None) -> dict[str, np.ndarray] | None:
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)[None]
        if self.buffer is None:  # before the first update the history holds the initial state at rest
            self.buffer = HistoryBuffer(self.policy.cfg, f(q), f(qd), self.inverse_dynamics, self.dt)
        self.buffer.push(f(q), f(qd), f(self.last_tau), f(q_ref))
        if not evaluate:
            return None
        with torch.no_grad():
            obs = self.buffer.observation(f(q), f(qd), f(q_ref), f(qd_ref), self.limits_t, None if q_ref_future is None else f(q_ref_future), self.extras)
            self.last_obs = obs
            out = self.policy.decode(self.policy(obs))
        self.n_updates += 1
        return {k: v[0].numpy() for k, v in out.items()}

    def base_term(self, q, qd_ref, qdd_ref) -> np.ndarray:
        """The model term of the PD base: gravity, or the full inverse-dynamics feedforward of the reference."""
        if self.feedforward and qdd_ref is not None:
            return self.nominal.mass_matrix(q) @ np.asarray(qdd_ref) + self.nominal.bias(q, np.asarray(qd_ref))
        return self.nominal.gravity(q)

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref, q_ref_future=None):
        if self.hold > 1:  # history at every step, decision every ``hold`` steps, output held in between (after the latency)
            out = self._network_update(q, qd, q_ref, qd_ref, evaluate=self.k % self.hold == 0, q_ref_future=q_ref_future)
            if out is not None:
                self.pending.append((self.k + self.latency, out))
        elif self.k % self.period == 0:
            self.pending.append((self.k + self.latency, self._network_update(q, qd, q_ref, qd_ref, q_ref_future=q_ref_future)))
        while self.pending and self.k >= self.pending[0][0]:
            self.held = self.pending.popleft()[1]
        self.k += 1
        held = self.held if self.held is not None else {k: v[0].numpy() for k, v in self.policy.neutral_output(1).items()}
        held = self.filter(held)
        base = self.base_term(q, qd_ref, qdd_ref)
        if self.kind == "learned_gains":
            tau = held["kp"] * (q_ref - q) + held["kd"] * (qd_ref - qd) + base
        elif self.kind == "pd_residual":
            tau = self.kp_shared * (q_ref - q) + self.kd_shared * (qd_ref - qd) + base + held["residual"]
        elif self.kind == "pd_gain_residual":
            tau = held["kp_scale"] * self.kp_shared * (q_ref - q) + held["kd_scale"] * self.kd_shared * (qd_ref - qd) + base + held["residual"]
        else:
            tau = held["torque"]
        self.last_tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
        return tau

    def frame(self, t, q, qd, q_ref, qd_ref, qdd_ref, q_ref_future=None) -> dict:
        """The impedance frame at a host instant (ADR 0011): the network decides now (the servo loop calls this
        once per host frame, so the policy's period and hold are in host frames and its compute latency is 0),
        and the fast law's gains and additive terms are returned for the servo to hold."""
        from llc.sim.servo import make_frame

        assert self.latency == 0 and self.hold == 1 and self.period == 1, "a policy run through a servo profile decides at every host frame"
        held = self._network_update(q, qd, q_ref, qd_ref, q_ref_future=q_ref_future)
        self.held = held = self.filter(held)
        self.k += 1
        base = self.base_term(q, qd_ref, qdd_ref)
        if self.kind == "learned_gains":
            return make_frame(q_ref, qd_ref, held["kp"], held["kd"], base)
        if self.kind == "pd_residual":
            return make_frame(q_ref, qd_ref, self.kp_shared, self.kd_shared, base + held["residual"])
        if self.kind == "pd_gain_residual":
            return make_frame(q_ref, qd_ref, held["kp_scale"] * self.kp_shared, held["kd_scale"] * self.kd_shared, base + held["residual"])
        return make_frame(q_ref, qd_ref, np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS), held["torque"])


def load_learned_controller(path, nominal: NominalModel, **options) -> LearnedController:
    policy, _ = load_policy(path)
    return LearnedController(policy, nominal, **options)
