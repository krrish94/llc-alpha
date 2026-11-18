"""Interface profiles and the servo loop (ADR 0011): P0 reproduces the torque interface exactly, the
model-based interface is exact at the frame's arrival for every profile, a learned controller through a
profile decides at the host rate, the oracle's frame reproduces its torque, and the torque-tracking
metrics read the realised torque against the command."""
import numpy as np

from llc.controllers.factory import build_controller
from llc.controllers.pd import PDFeedforward, PDGravity
from llc.sim.nominal import NominalModel
from llc.sim.panda import HOME, NUM_JOINTS, TORQUE_LIMITS
from llc.sim.servo import ModelInterface, ServoLoop, frame_law, get_profile, make_frame

KP, KD = np.array([400.0, 400.0, 300.0, 300.0, 100.0, 60.0, 30.0]), np.array([40.0, 40.0, 30.0, 30.0, 10.0, 6.0, 3.0])


def _state(k: int):
    q = HOME + 0.02 * np.sin(0.1 * k + np.arange(7))
    qd = 0.3 * np.cos(0.1 * k + np.arange(7))
    return q, qd


def test_p0_reproduces_the_torque_interface():
    nominal = NominalModel()
    pd = PDFeedforward(KP, KD, nominal)
    loop = ServoLoop(PDFeedforward(KP, KD, nominal), get_profile("p0"), ModelInterface(get_profile("p0"), KP, KD))
    for k in range(20):
        q, qd = _state(k)
        q_ref, qd_ref, qdd_ref = HOME + 0.01 * k, np.full(7, 0.1), np.full(7, 0.5)
        assert np.allclose(loop(k * 1e-3, q, qd, q_ref, qd_ref, qdd_ref), np.clip(pd(k * 1e-3, q, qd, q_ref, qd_ref, qdd_ref), -TORQUE_LIMITS, TORQUE_LIMITS))
        assert np.allclose(loop.tau_command, loop.last_tau)


def test_model_interface_exact_at_arrival_for_every_profile():
    """A frame sent at a measured state, received after the bus delay, is realised exactly when the robot
    moves as predicted (uniform velocity), up to the profile's quantisation."""
    nominal = NominalModel()
    for name in ("p1", "p2", "p3"):
        p = get_profile(name, delay=0.0)
        iface = ModelInterface(p, KP, KD)
        q, qd = HOME + 0.05, np.full(7, 0.2)
        q_ref, qd_ref = HOME.copy(), np.zeros(7)
        upper = make_frame(q_ref, qd_ref, KP, KD, nominal.gravity(q))
        servo = iface(upper, q, qd)
        tau_servo = frame_law(servo, q, qd)
        tau_upper = frame_law(upper, q, qd)
        tol = 0.25 * KP * p.goal_res + 0.25 * KD * p.vel_res + 1e-6  # a quantisation step of the goal or the velocity
        assert np.all(np.abs(tau_servo - tau_upper) <= tol + 0.02 * np.abs(tau_upper)), (name, tau_servo - tau_upper)
        if p.gains == "fixed":
            assert np.allclose(servo["kp"], KP) and np.allclose(servo["kd"], KD)


def test_torque_only_command_is_held_with_the_servo_spring():
    p = get_profile("p2")
    iface = ModelInterface(p, KP, KD)
    q, qd = HOME.copy(), np.zeros(7)
    tau = np.array([5.0, -5.0, 2.0, 0.0, 1.0, -1.0, 0.5])
    servo = iface(make_frame(q, qd, np.zeros(7), np.zeros(7), tau), q, qd)
    assert np.allclose(servo["kp"], KP, rtol=1e-3)  # the gain channel's 12-bit quantisation
    assert np.all(np.abs(frame_law(servo, q, qd) - tau) <= 0.5 * KP * p.goal_res + 1e-3 * np.abs(tau) + 1e-9)


def test_servo_loop_holds_frames_and_delays_them():
    nominal = NominalModel()
    p = get_profile("p2", period=0.01, delay=0.002)
    loop = ServoLoop(PDGravity(KP, KD, nominal), p, ModelInterface(p, KP, KD))
    q, qd = HOME + 0.03, np.zeros(7)
    taus = []
    for k in range(25):
        q_ref = HOME + (0.05 if k >= 10 else 0.0)
        taus.append(loop(k * 1e-3, q, qd, q_ref, np.zeros(7), np.zeros(7)).copy())
    taus = np.array(taus)
    assert loop.n_frames == 3  # frames at k = 0, 10, 20
    assert np.allclose(taus[2], taus[11])  # the frame of k = 10 (new reference) takes effect at k = 12
    assert not np.allclose(taus[11], taus[12])
    assert np.allclose(taus[12], taus[21])


def test_learned_controller_through_a_profile_decides_per_frame():
    import torch

    from llc.learning.policy import Policy, PolicyConfig, save_policy

    nominal = NominalModel()
    cfg = PolicyConfig(kind="pd_gain_residual", rate_hz=250.0, latency_s=0.0, history=4, history_stride=1, history_dense=0, kp_shared=tuple(KP), kd_shared=tuple(KD), feedforward=True, inertia_diag=tuple(np.ones(7)))
    pol = Policy(cfg)
    path = "/tmp/llc_test_servo_policy.pt"
    save_policy(path, pol)
    spec = {"kind": "servo", "profile": "p1", "inner": {"kind": "learned", "path": path}, "kp": KP.tolist(), "kd": KD.tolist()}
    loop = build_controller(spec, nominal)
    pd = PDFeedforward(KP, KD, nominal)
    for k in range(12):
        q, qd = _state(k)
        q_ref, qd_ref, qdd_ref = HOME.copy(), np.zeros(7), np.zeros(7)
        tau = loop(k * 1e-3, q, qd, q_ref, qd_ref, qdd_ref)
    assert loop.inner.n_updates == 3 and loop.inner.period == 1 and loop.inner.latency == 0  # 250 Hz: frames at k = 0, 4, 8
    # a neutral policy (gain multipliers 1, zero residual) through P1 reproduces the feedforward PD up to quantisation
    q, qd = _state(11)
    tau_pd = pd(0.011, q, qd, HOME, np.zeros(7), np.zeros(7))
    assert np.all(np.abs(tau - tau_pd) <= KP * 25.0 / 65536 + KD * 60.0 / 4096 + 2 * TORQUE_LIMITS / 4096 + 1e-6)


def test_oracle_frame_reproduces_its_torque():
    from llc.eval.cases_v2 import make_case_v2

    nominal = NominalModel()
    case = make_case_v2("blocked", "val", 0, nominal)
    base = {"kind": "pd_feedforward", "kp": KP.tolist(), "kd": KD.tolist()}
    oracle = build_controller({"kind": "oracle", "base": base}, nominal, case)
    oracle2 = build_controller({"kind": "oracle", "base": base}, nominal, case)
    ref = case.reference
    k_in = int(np.where(oracle.depth_ref > 0)[0][0]) + 50
    q = ref.q[k_in] + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for k in range(k_in + 1):  # the oracles must advance together
        tau = oracle(ref.t[k], ref.q[k], ref.qd[k], ref.q[k], ref.qd[k], ref.qdd[k])
        fr = oracle2.frame(ref.t[k], ref.q[k], ref.qd[k], ref.q[k], ref.qd[k], ref.qdd[k])
    assert np.allclose(frame_law(fr, ref.q[k_in], ref.qd[k_in]), tau)


def test_oracle_through_a_profile_keeps_its_contact_law():
    """The oracle indexes the reference by time: called once per 4 ms frame it bounds the blocked force as at 1 kHz."""
    from llc.eval.cases_v2 import make_case_v2
    from llc.eval.metrics import compute_metrics
    from llc.eval.rollout import rollout
    from llc.sim.panda import PandaSim

    nominal = NominalModel()
    case = make_case_v2("blocked", "val", 1, nominal)
    base = {"kind": "pd_feedforward", "kp": KP.tolist(), "kd": KD.tolist()}
    sim = PandaSim(case.perturbation)
    exc = {}
    for profile in ("p0", "p1"):
        spec = {"kind": "servo", "profile": profile, "inner": {"kind": "oracle", "base": base}, "kp": KP.tolist(), "kd": KD.tolist()}
        ep = rollout(sim, build_controller(spec, nominal, case), case)
        exc[profile] = compute_metrics(ep, case)["excess_force_cap"]
    pd_exc = compute_metrics(rollout(sim, build_controller(base, nominal, case), case), case)["excess_force_cap"]
    assert exc["p1"] < 0.5 * pd_exc + 2.0 and abs(exc["p1"] - exc["p0"]) < 0.3 * pd_exc + 2.0, (exc, pd_exc)


def test_torque_tracking_metrics():
    from llc.eval.cases_v2 import make_case_v2
    from llc.eval.metrics import compute_metrics
    from llc.eval.rollout import rollout
    from llc.sim.panda import PandaSim

    nominal = NominalModel()
    case = make_case_v2("torque", "test", 2, nominal)  # index 2: the chirp program
    assert case.meta["program"]["kind"] == "chirp"
    base = {"kind": "pd_feedforward", "kp": KP.tolist(), "kd": KD.tolist()}
    sim = PandaSim(case.perturbation)
    for profile, max_rmse in (("p0", 1e-9), ("p1", 0.05)):
        spec = {"kind": "servo", "profile": profile, "inner": {"kind": "torque_program", "base": base}, "kp": KP.tolist(), "kd": KD.tolist()}
        ep = rollout(sim, build_controller(spec, nominal, case), case)
        m = compute_metrics(ep, case)
        assert np.isfinite(m["tau_track_rmse"]) and m["tau_track_rmse"] <= max_rmse, (profile, m["tau_track_rmse"])
        assert np.isfinite(m["tau_track_bandwidth"]) and np.isfinite(m["composite_cap"])
    assert m["tau_track_delay"] >= 0.0
