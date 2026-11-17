"""Feedforward PD base of the learned controllers (ADR 0010, round 2): the fast loop adds the nominal
inverse-dynamics feedforward of the reference instead of gravity alone, in the simulator controller,
the torch training rollouts and the oracle teacher."""
import json

import numpy as np
import torch

from llc.controllers.learned import LearnedController
from llc.controllers.oracle import oracle_for_case
from llc.controllers.pd import PDFeedforward, PDGravity
from llc.dynamics.rigid_body import RigidBodyModel
from llc.eval.cases_v2 import make_case_v2
from llc.learning.actor_critic import reference_accel
from llc.learning.policy import Policy, PolicyConfig
from llc.sim.nominal import NominalModel
from llc.sim.panda import HOME, load_model


def _gains():
    return json.load(open("runs/d1/gains_shared.json"))["spec"]


def _state(rng):
    q = HOME + rng.normal(0.0, 0.2, 7)
    return q, rng.normal(0.0, 0.5, 7), q + rng.normal(0.0, 0.05, 7), rng.normal(0.0, 0.8, 7), rng.normal(0.0, 5.0, 7)


def test_learned_controller_with_the_feedforward_base_reproduces_pd_feedforward():
    nominal, rng, g = NominalModel(), np.random.default_rng(0), _gains()
    cfg = PolicyConfig(kind="pd_residual", kp_shared=tuple(g["kp"]), kd_shared=tuple(g["kd"]), rate_hz=1000.0, latency_s=0.0, feedforward=True)
    ctrl = LearnedController(Policy(cfg), nominal)
    ref = PDFeedforward(np.array(g["kp"]), np.array(g["kd"]), nominal)
    ctrl.reset()
    for k in range(3):
        q, qd, q_ref, qd_ref, qdd_ref = _state(rng)
        tau = ctrl(k * 1e-3, q, qd, q_ref, qd_ref, qdd_ref)
        assert np.allclose(tau - ctrl.filter(ctrl.held)["residual"], ref(k * 1e-3, q, qd, q_ref, qd_ref, qdd_ref), atol=1e-5)
        assert np.allclose(ctrl.base_term(q, qd_ref, qdd_ref), ref.base_term(q, qd_ref, qdd_ref))
    plain = LearnedController(Policy(PolicyConfig(kind="pd_residual", kp_shared=tuple(g["kp"]), kd_shared=tuple(g["kd"]), rate_hz=1000.0, latency_s=0.0)), nominal)
    q, qd, q_ref, qd_ref, qdd_ref = _state(rng)
    assert np.allclose(plain.base_term(q, qd_ref, qdd_ref), nominal.gravity(q))  # without the flag the base term is gravity, as before


def test_torch_feedforward_matches_the_nominal_model():
    nominal, rng = NominalModel(), np.random.default_rng(1)
    model = RigidBodyModel(load_model(), dtype=torch.float64)
    for _ in range(3):
        q, _, _, qd_ref, qdd_ref = _state(rng)
        want = nominal.mass_matrix(q) @ qdd_ref + nominal.bias(q, qd_ref)
        got = model.inverse_dynamics(torch.as_tensor(q)[None], torch.as_tensor(qd_ref)[None], torch.as_tensor(qdd_ref)[None])[0].numpy()
        assert np.allclose(got, want, atol=1e-6, rtol=1e-6)


def test_reference_accel_is_a_clipped_central_difference():
    dt, t = 1e-3, np.arange(0, 1.0, 1e-3)
    qd = torch.as_tensor(np.stack([np.sin(2 * np.pi * 2.0 * t + j) for j in range(7)], 1)[None], dtype=torch.float32)
    qdd = reference_accel(qd, dt)
    want = torch.as_tensor(np.stack([2 * np.pi * 2.0 * np.cos(2 * np.pi * 2.0 * t + j) for j in range(7)], 1)[None], dtype=torch.float32)
    assert torch.allclose(qdd[:, 1:-1], want[:, 1:-1], atol=2e-3)
    assert float(qdd[:, 0].abs().max()) == 0.0 and float(qdd[:, -1].abs().max()) == 0.0
    step = torch.zeros(1, 50, 7); step[:, 25:] = 1.0  # a held-target jump of 1 rad/s in one step: 500 rad/s^2 raw, clipped
    assert float(reference_accel(step, dt).abs().max()) == 50.0


def test_oracle_removes_the_model_term_of_its_base():
    """Around a feedforward base the oracle projects only the PD feedback into a tool force, so its
    output differs from the oracle around the gravity base exactly by the feedforward term."""
    nominal, g = NominalModel(), _gains()
    case = make_case_v2("contact2", "val", 0, nominal)
    ref = case.reference
    oracles = [oracle_for_case(base, nominal, case) for base in (PDGravity(np.array(g["kp"]), np.array(g["kd"]), nominal), PDFeedforward(np.array(g["kp"]), np.array(g["kd"]), nominal))]
    k_c = int(round(case.meta["t_contact_nominal"] / 1e-3))
    pressed = 0
    for k in range(k_c + 150):  # tracking the reference exactly: the tool is inside the surface once the reference is
        q, qd = ref.q[k], ref.qd[k]
        taus = [o(ref.t[k], q, qd, ref.q[k], ref.qd[k], ref.qdd[k]) for o in oracles]
        pressed += oracles[1].push > 0.0
        diff = PDFeedforward(np.array(g["kp"]), np.array(g["kd"]), nominal).base_term(q, ref.qd[k], ref.qdd[k]) - nominal.gravity(q)
        assert np.allclose(taus[1] - taus[0], diff, atol=1e-6)
    assert pressed > 50 and abs(oracles[0].push - oracles[1].push) < 1e-9


def test_ideal_label_is_the_command_on_the_nominal_plant_under_a_pulse():
    """On the nominal plant the ideal-robot label equals the command whatever external load acts (the ideal robot
    feels the same load), so a tool-force pulse enters as an external torque and is not cancelled."""
    from llc.learning.ideal import ideal_label
    from llc.sim.instances import DynamicsPerturbation
    from llc.sim.panda import PandaSim

    nominal, rng = NominalModel(), np.random.default_rng(3)
    sim = PandaSim(DynamicsPerturbation(), seed=0)
    q, qd, _, _, _ = _state(rng)
    sim.reset(q, qd)
    from llc.sim.panda import TORQUE_LIMITS

    tau_cmd = rng.uniform(-0.5, 0.5, 7) * TORQUE_LIMITS  # within the actuator limits (the label is clipped to them)
    assert np.allclose(ideal_label(sim, nominal, q, qd, tau_cmd, np.zeros(7)), tau_cmd, atol=1e-6)
    assert np.allclose(ideal_label(sim, nominal, q, qd, tau_cmd, rng.normal(0.0, 5.0, 7)), tau_cmd, atol=1e-6)


def test_two_teacher_labels_cover_free_space_and_contact():
    """With the ideal free-space teacher, free-space cases return ideal-robot labels of the student's own drive and
    contact cases the oracle's, in the same field layout."""
    from llc.eval.parallel import CaseSpec
    from llc.learning.distill import FIELDS, collect_labels
    from llc.learning.policy import Policy, save_policy
    import tempfile

    g = _gains()
    cfg = PolicyConfig(kind="pd_residual", kp_shared=tuple(g["kp"]), kd_shared=tuple(g["kd"]), rate_hz=1000.0, latency_s=0.0, feedforward=True, hold_steps=10)
    with tempfile.TemporaryDirectory() as d:
        save_policy(f"{d}/s.pt", Policy(cfg))
        teacher = {"kind": "oracle", "base": {"kind": "learned", "path": f"{d}/s.pt"}, "free_teacher": "ideal"}
        batches = collect_labels(f"{d}/s.pt", teacher, [CaseSpec("hold", "val", 0, protocol="v2"), CaseSpec("contact2", "val", 0, protocol="v2")], [False, True], workers=1)
    for b in batches:
        assert set(FIELDS) <= set(b) and b["obs"].shape[0] > 50 and b["tau_teacher"].shape == b["gravity"].shape
    assert str(batches[0]["regime"][0]) == "hold" and str(batches[1]["regime"][0]) == "contact2"
