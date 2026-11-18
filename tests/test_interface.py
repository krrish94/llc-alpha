"""The learned interface (ADR 0011): its torch law agrees with the numpy model-based interface, a zero head
reproduces the model-based interface exactly through the servo loop, and the training rollout runs."""
import numpy as np
import torch

from llc.learning.interface import InterfaceConfig, InterfacePolicy, LearnedInterface, frame_law_torch, model_interface_torch, save_interface
from llc.sim.nominal import NominalModel
from llc.sim.panda import HOME, TORQUE_LIMITS
from llc.sim.servo import ModelInterface, ServoLoop, frame_law, get_profile, make_frame

KP, KD = np.array([400.0, 400.0, 300.0, 300.0, 100.0, 60.0, 30.0]), np.array([40.0, 40.0, 30.0, 30.0, 10.0, 6.0, 3.0])


def test_torch_interface_matches_numpy():
    rng = np.random.default_rng(0)
    for name in ("p1", "p2", "p3"):
        p = get_profile(name)
        mi = ModelInterface(p, KP, KD)
        for _ in range(5):
            q, qd = HOME + rng.normal(0, 0.1, 7), rng.normal(0, 0.5, 7)
            fr = make_frame(q + rng.normal(0, 0.05, 7), rng.normal(0, 0.3, 7), KP * rng.uniform(0.5, 1.5, 7), KD * rng.uniform(0.5, 1.5, 7), rng.normal(0, 5, 7))
            a = mi(fr, q, qd)
            f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)[None]
            b = model_interface_torch(p, f(KP)[0], f(KD)[0], {k: f(v) for k, v in fr.items()}, f(q), f(qd))
            for key in a:
                assert np.allclose(a[key], b[key][0].numpy(), rtol=1e-4, atol=1e-4), (name, key, a[key], b[key][0].numpy())


def test_zero_head_reproduces_model_interface_through_the_loop():
    from llc.controllers.pd import PDFeedforward

    nominal = NominalModel()
    for name in ("p1", "p2"):
        p = get_profile(name)
        cfg = InterfaceConfig(name, tuple(KP), tuple(KD), history=4, hidden=32, layers=2)
        pol = InterfacePolicy(cfg)
        a = ServoLoop(PDFeedforward(KP, KD, nominal), p, ModelInterface(p, KP, KD))
        b = ServoLoop(PDFeedforward(KP, KD, nominal), p, LearnedInterface(pol, p, KP, KD))
        for k in range(30):
            q = HOME + 0.02 * np.sin(0.1 * k + np.arange(7)); qd = 0.3 * np.cos(0.1 * k + np.arange(7))
            ta = a(k * 1e-3, q, qd, HOME, np.zeros(7), np.zeros(7)); tb = b(k * 1e-3, q, qd, HOME, np.zeros(7), np.zeros(7))
            assert np.allclose(ta, tb, atol=1e-4), (name, k, ta - tb)
        assert b.interface.hist is not None and len(b.interface.hist) == 4


def test_training_rollout_runs_and_backpropagates(tmp_path):
    from experiments.m11_interface import rollout_interface, sample_programs
    from llc.dynamics.rigid_body import RigidBodyModel
    from llc.learning.mjgrad import MjBatch
    from llc.sim.instances import sample_instance
    from llc.sim.panda import DynamicsPerturbation, load_model

    rng = np.random.default_rng(0)
    cfg = InterfaceConfig("p2", tuple(KP), tuple(KD), history=4, hidden=32, layers=2)
    pol = InterfacePolicy(cfg)
    with torch.no_grad():
        pol.head.weight.normal_(0, 0.01)
    model = RigidBodyModel(load_model(), dtype=torch.float32)
    B, T = 2, 60
    batch = MjBatch([DynamicsPerturbation(), sample_instance(rng)], threads=1)
    q_ref = torch.as_tensor(np.repeat(HOME[None, None], T, 1).repeat(B, 0).astype(np.float32)); qd_ref = torch.zeros(B, T, 7)
    programs = sample_programs(["hold", "hold"], rng)
    programs[0] = {"kind": "step", "joint": 1, "amplitude": 0.2, "t_start": 0.01}
    res = rollout_interface(pol, batch, model, q_ref, qd_ref, np.repeat(HOME[None], B, 0), np.zeros((B, 7)), programs, train=True, tbptt=30)
    assert np.isfinite(res["loss"]) and res["tau"].shape == (B, T, 7)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in pol.parameters())
    save_interface(tmp_path / "i.pt", pol)
