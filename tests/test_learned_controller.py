"""Timing and consistency of the learned controller: update period, latency (causality), history
content, and agreement between the simulator-side fast loop and the torch rollout's fast loop."""
import numpy as np
import torch

from llc.controllers.learned import LearnedController
from llc.learning.policy import Policy, PolicyConfig, encode_obs
from llc.sim.nominal import NominalModel
from llc.sim.panda import HOME


def _policy(kind: str) -> Policy:
    torch.manual_seed(0)
    cfg = PolicyConfig(kind=kind, inertia_diag=tuple(np.ones(7)), kp_shared=tuple(np.full(7, 100.0)), kd_shared=tuple(np.full(7, 10.0)))
    pol = Policy(cfg)
    with torch.no_grad():
        pol.head.weight.normal_(0, 0.5)  # make outputs depend on the observation
    return pol


def test_update_period_and_latency():
    nominal = NominalModel()
    ctrl = LearnedController(_policy("neural_torque"), nominal, dt=1e-3)
    assert ctrl.period == 10 and ctrl.latency == 5
    q, qd = HOME.copy(), np.zeros(7)
    taus = []
    for k in range(25):
        q_ref = HOME + 0.1 * (k >= 3)  # the reference changes at step 3, after the first network update
        taus.append(ctrl(k * 1e-3, q, qd, q_ref, qd, qd).copy())
    taus = np.array(taus)
    # steps 0..4: neutral output (nothing has taken effect yet); step 5..14: output of the update at k=0,
    # which saw q_ref = HOME; the k=10 update sees the new reference and takes effect at k=15.
    assert np.allclose(taus[0], taus[4])
    assert np.allclose(taus[5], taus[14])
    assert not np.allclose(taus[14], taus[15])
    assert ctrl.n_updates == 3


def test_history_holds_past_samples():
    nominal = NominalModel()
    ctrl = LearnedController(_policy("learned_gains"), nominal, dt=1e-3)
    qd = np.zeros(7)
    for k in range(31):
        q = HOME + 0.001 * k
        ctrl(k * 1e-3, q, qd, HOME, qd, qd)
    # updates at k = 0, 10, 20, 30 -> the last four history rows hold those positions
    assert np.allclose(ctrl.hist_q[-1], HOME + 0.030)
    assert np.allclose(ctrl.hist_q[-2], HOME + 0.020)
    assert np.allclose(ctrl.hist_q[-4], HOME)


def test_fast_loop_matches_torch_fast_torque():
    nominal = NominalModel()
    for kind in ("learned_gains", "pd_residual", "neural_torque"):
        pol = _policy(kind)
        ctrl = LearnedController(pol, nominal, dt=1e-3)
        q, qd, q_ref, qd_ref = HOME + 0.05, np.full(7, 0.1), HOME.copy(), np.zeros(7)
        for k in range(6):  # first output takes effect at k = 5
            tau_np = ctrl(k * 1e-3, q, qd, q_ref, qd_ref, qd_ref)
        f = lambda x: torch.as_tensor(x, dtype=torch.float32)[None]
        held = {k: f(v) for k, v in ctrl.held.items()}
        tau_t = pol.fast_torque(held, f(q), f(qd), f(q_ref), f(qd_ref), f(nominal.gravity(q)))[0].numpy()
        assert np.allclose(tau_np, tau_t, atol=1e-4), kind


def test_hold_policy_honours_a_compute_latency():
    """A 10 ms decision policy with a 5 ms latency applies each decision 5 steps after it is made (ADR 0010)."""
    import numpy as np
    import torch
    from dataclasses import replace

    from llc.controllers.learned import LearnedController
    from llc.learning.policy import load_policy
    from llc.sim.nominal import NominalModel

    policy, _ = load_policy("runs/d7/distill_gain_ext-s0/best.pt")
    assert policy.cfg.hold_steps == 10
    nominal = NominalModel()
    q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]); qd = np.zeros(7)
    taus = {}
    for lat in (0.0, 0.005):
        ctrl = LearnedController(policy, nominal, latency_s=lat)
        ctrl.reset()
        out = []
        for k in range(30):
            q_ref = q + 0.02 * np.sin(0.3 * k)  # a moving target so that decisions differ
            out.append(ctrl(k * 1e-3, q, qd, q_ref, np.zeros(7), np.zeros(7)).copy())
        taus[lat] = np.array(out)
    # without latency the first decision acts at step 0; with 5 ms latency the neutral output is applied until step 5
    assert not np.allclose(taus[0.0][1], taus[0.005][1])
    assert np.allclose(taus[0.0][5], taus[0.005][5], atol=1e-6) or np.allclose(taus[0.0][0], taus[0.005][5], atol=1e-6)


def test_latency_queue_applies_the_first_decision_after_the_latency_for_1khz_and_10ms_policies():
    """Decisions in flight are queued (ADR 0010): with a 5 ms latency the held output is neutral for five
    steps and the first decision takes effect at step 5, for a 1 kHz policy and for a 10 ms one."""
    import numpy as np

    from llc.controllers.learned import LearnedController
    from llc.learning.policy import load_policy
    from llc.sim.nominal import NominalModel

    nominal = NominalModel()
    q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]); qd = np.zeros(7)
    for path in ("runs/m9/pd_residual_ac-s0/best.pt", "runs/m9/distill_gain_ext-s0/best.pt"):
        policy, _ = load_policy(path)
        held = {}
        for lat in (0.0, 0.005):
            ctrl = LearnedController(policy, nominal, latency_s=lat); ctrl.reset(); hs = []
            for k in range(8):
                ctrl(k * 1e-3, q, qd, q + 0.02 * np.sin(0.3 * k), np.zeros(7), np.zeros(7)); hs.append(None if ctrl.held is None else ctrl.held["residual"].copy())
            held[lat] = hs
        assert all(h is None for h in held[0.005][:5]) and np.allclose(held[0.0][0], held[0.005][5])
