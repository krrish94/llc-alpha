"""The DC-motor layer (ADR 0011 amendment 3): nominal parameters are transparent, the envelope limits driving
torque only, the lag and friction act as written, the firmware's feedback reaches the upper layers."""
import numpy as np

from llc.sim.motor import MotorLayer, MotorParams, nominal, params_for, sample
from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS


class _Const:
    def __init__(self, tau):
        self.tau = np.asarray(tau, float)
        self.last_tau = np.zeros(NUM_JOINTS)

    def reset(self):
        pass

    def __call__(self, t, q, qd, q_ref, qd_ref, qdd_ref, q_ref_future=None):
        return self.tau.copy()


def _call(layer, qd):
    z = np.zeros(NUM_JOINTS)
    return layer(0.0, z, np.asarray(qd, float), z, z, z)


def test_nominal_is_transparent():
    tau = 0.5 * TORQUE_LIMITS
    layer = MotorLayer(_Const(tau), nominal())
    out = _call(layer, 0.9 * np.ones(NUM_JOINTS))
    assert np.allclose(out, tau)
    assert np.allclose(layer.inner.last_tau, tau)


def test_envelope_limits_driving_torque_only():
    m = MotorParams(np.ones(NUM_JOINTS), 2.0 * TORQUE_LIMITS, np.full(NUM_JOINTS, 2.0), np.zeros(NUM_JOINTS), np.zeros(NUM_JOINTS), 0.0, 0.0)
    layer = MotorLayer(_Const(TORQUE_LIMITS), m)
    out = _call(layer, np.full(NUM_JOINTS, 1.5))  # driving at 75% of the no-load speed: the envelope is 2 (1 - 0.75) = 0.5 of the peak
    assert np.allclose(out, 0.5 * TORQUE_LIMITS)
    out = _call(layer, np.full(NUM_JOINTS, -1.5))  # braking: not limited by the back-EMF
    assert np.allclose(out, TORQUE_LIMITS)


def test_lag_and_friction_and_feedback():
    m = MotorParams(np.full(NUM_JOINTS, 1.1), 1e3 * TORQUE_LIMITS, np.full(NUM_JOINTS, 1e3), np.full(NUM_JOINTS, 0.1), np.full(NUM_JOINTS, 0.2), 1e-3, 0.0)
    layer = MotorLayer(_Const(np.ones(NUM_JOINTS)), m)
    out = _call(layer, np.zeros(NUM_JOINTS))
    assert np.allclose(out, 1.1 * 0.5)  # one step of a 1 ms lag at 1 ms: half way to the commanded 1.1
    out = _call(layer, np.ones(NUM_JOINTS))
    assert np.allclose(out, 0.825 - 0.1 - 0.2 * np.tanh(20.0))  # the lag at 0.825 of 1.1, viscous 0.1 and Coulomb 0.2 at 1 rad/s
    assert np.allclose(layer.inner.last_tau * 1.1, layer.tau_f)  # the feedback is the firmware's current times its own constant


def test_random_draw_is_deterministic_per_seed():
    a, b, c = params_for("random", 5), params_for("random", 5), params_for("random", 6)
    assert np.allclose(a.kt_scale, b.kt_scale) and not np.allclose(a.kt_scale, c.kt_scale)
    s = sample(np.random.default_rng(0))
    assert s.tau_stall.min() >= 1.5 * TORQUE_LIMITS.min() and 0.0 <= s.lag <= 2e-3
