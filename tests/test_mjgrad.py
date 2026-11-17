"""MjBatch (differentiable MuJoCo stepping by finite differences): gradients against the analytic
rigid-body model in free space and against brute-force differences in contact; state consistency,
batch independence and the torque clamp."""

import copy

import mujoco
import numpy as np
import pytest
import torch

from llc.dynamics.rigid_body import RigidBodyModel
from llc.learning.mjgrad import MjBatch
from llc.sim.nominal import NominalModel
from llc.sim.panda import HOME, TORQUE_LIMITS, DynamicsPerturbation, PandaSim, load_model

DT = 0.001


def nominal_batch(B, **kw):
    return MjBatch([DynamicsPerturbation()] * B, **kw)


def free_states(B, rng):
    """Collision-free states near HOME: small position offsets, qd in [-1, 1]."""
    return HOME + rng.uniform(-0.1, 0.1, (B, 7)), rng.uniform(-1.0, 1.0, (B, 7))


def rollout_loss(step, q, qd, theta):
    """Sum over steps of |q_t|^2 for the torque sequence ``theta`` (T, B, 7) under ``step(q, qd, tau)``."""
    loss = 0.0
    for t in range(theta.shape[0]):
        q, qd = step(q, qd, theta[t])
        loss = loss + (q**2).sum()
    return loss


@pytest.mark.parametrize("centered", [False, True])
def test_free_space_gradient_matches_analytic_model(centered):
    B, T = 3, 50
    rng = np.random.default_rng(0)
    q0, qd0 = free_states(B, rng)
    theta0 = rng.uniform(-10.0, 10.0, (T, B, 7))

    rbm = RigidBodyModel(load_model(), dtype=torch.float64)
    theta_a = torch.tensor(theta0, requires_grad=True)
    rollout_loss(lambda q, qd, tau: rbm.step(q, qd, tau, DT), torch.tensor(q0), torch.tensor(qd0), theta_a).backward()

    batch = nominal_batch(B, centered=centered)
    batch.reset(q0, qd0)
    theta = torch.tensor(theta0, dtype=torch.float32, requires_grad=True)
    rollout_loss(lambda q, qd, tau: batch.step(q, qd, tau)[:2], *batch.state(), theta).backward()

    g, g_a = theta.grad.double(), theta_a.grad
    rel = float((g - g_a).norm() / g_a.norm())
    assert rel < 1e-2, f"relative gradient error {rel:.3e}"
    assert rel < 1e-4  # measured 1.8e-6 forward / 9e-7 centered; FD noise is roundoff-dominated


def test_contact_gradient_vs_brute_force():
    """Tool 3 cm above the table, torque = g(q) + theta * J0^T (0, 0, -1): the tip is pushed into the
    table within the horizon.  d loss / d theta from autograd vs centred differences over the rollout."""
    T, theta0, h = 300, 10.0, 1e-3
    nom = NominalModel()
    q0, err = nom.ik(np.array([0.45, 0.0, 0.20]), HOME)
    assert err < 1e-3 and abs(nom.tool_pos(q0)[2] - 0.20) < 1e-3  # table top at z = 0.15, tip radius 2 cm
    u = torch.as_tensor(nom.tool_jacobian(q0)[0].T @ np.array([0.0, 0.0, -1.0]), dtype=torch.float32)
    rbm = RigidBodyModel(load_model(), dtype=torch.float32)  # differentiable gravity compensation

    def rollout(theta):
        batch = nominal_batch(1)
        batch.reset(q0[None], np.zeros((1, 7)))
        q, qd = batch.state()
        loss, touches = torch.zeros((), dtype=torch.float64), []
        for _ in range(T):
            tau = rbm.bias(q, torch.zeros_like(qd)) + theta * u
            q, qd, touch, _ = batch.step(q, qd, tau)
            loss = loss + (touch.double() ** 2).sum() + (q.double() ** 2).sum()
            touches.append(float(touch.detach()))
        return loss, np.array(touches)

    theta = torch.tensor(theta0, requires_grad=True)
    loss, touches = rollout(theta)
    loss.backward()
    assert 0 < np.argmax(touches > 0) < T - 50 and touches.max() > 50.0  # contact well within the horizon
    with torch.no_grad():
        g_fd = float(rollout(torch.tensor(theta0 + h))[0] - rollout(torch.tensor(theta0 - h))[0]) / (2 * h)
    g_auto = float(theta.grad)
    rel = abs(g_auto - g_fd) / abs(g_fd)
    assert rel < 0.1, f"autograd {g_auto:.1f} vs brute force {g_fd:.1f}: {rel:.3e}"
    assert rel < 1e-2  # measured 9e-4 (an impact without a bounce; see the module report for bouncing cases)


def test_state_consistency():
    B = 2
    rng = np.random.default_rng(1)
    q0, qd0 = free_states(B, rng)
    tau0 = rng.uniform(-5.0, 5.0, (B, 7))

    # mjd_transitionFD followed by mj_step equals mj_step alone (the FD restores the state).
    sim = PandaSim()
    sim.reset(q0[0], qd0[0])
    data_ref = copy.copy(sim.data)
    data_ref.ctrl[:] = tau0[0]
    mujoco.mj_step(sim.model, data_ref)
    sim.data.ctrl[:] = tau0[0]
    m = sim.model
    mats = np.zeros((2 * m.nv, 2 * m.nv)), np.zeros((2 * m.nv, m.nu)), np.zeros((1, 2 * m.nv)), np.zeros((1, m.nu))
    mujoco.mjd_transitionFD(m, sim.data, 1e-6, False, *mats)
    mujoco.mj_step(m, sim.data)
    assert np.abs(sim.data.qpos - data_ref.qpos).max() < 1e-12
    assert np.abs(sim.data.qvel - data_ref.qvel).max() < 1e-12
    assert np.abs(sim.data.qacc_warmstart - data_ref.qacc_warmstart).max() < 1e-12

    # A differentiated batch rollout reproduces plain PandaSim stepping exactly, and the returned
    # tensors are the simulators' state.
    sims = [PandaSim() for _ in range(B)]
    batch = nominal_batch(B)
    batch.reset(q0, qd0)
    q, qd = batch.state()
    tau = torch.tensor(tau0, dtype=torch.float32, requires_grad=True)
    for i, s in enumerate(sims):
        s.reset(q0[i], qd0[i])
    for _ in range(10):
        q, qd, touch, tool = batch.step(q, qd, tau)
        for i, s in enumerate(sims):
            s.step(tau0[i].astype(np.float32))
            assert np.abs(s.q - batch.sims[i].q).max() < 1e-12
            assert np.abs(s.qd - batch.sims[i].qd).max() < 1e-12
            assert np.array_equal(q[i].detach().numpy(), batch.sims[i].q.astype(np.float32))
            assert np.array_equal(qd[i].detach().numpy(), batch.sims[i].qd.astype(np.float32))
            assert np.array_equal(tool[i], s.tool_pos)
    (q**2).sum().backward()
    assert torch.isfinite(tau.grad).all()
    with pytest.raises(ValueError):  # a stale state is rejected
        batch.step(*batch.state()[::-1], tau)


def test_batch_independence():
    B = 3
    rng = np.random.default_rng(2)
    q0, qd0 = free_states(B, rng)
    q0[:], qd0[:] = q0[0], qd0[0]  # identical initial states, so only the torque distinguishes the elements
    tau0 = torch.tensor(rng.uniform(-5.0, 5.0, (B, 7)), dtype=torch.float32)

    def run(tau):
        batch = nominal_batch(B)
        batch.reset(q0, qd0)
        tau = tau.clone().requires_grad_(True)
        q, qd = batch.state()
        for _ in range(5):
            q, qd, touch, _ = batch.step(q, qd, tau)
        per_element = (q**2).sum(1) + (qd**2).sum(1) + touch**2
        return tau, q, qd, per_element

    tau, q, qd, per_element = run(tau0)
    (grad_0,) = torch.autograd.grad(per_element[0], tau, retain_graph=True)
    assert grad_0[0].abs().sum() > 0 and grad_0[1:].abs().sum() == 0  # no cross-element gradient
    per_element.sum().backward()
    grad = tau.grad.clone()

    tau1 = tau0.clone()
    tau1[1] += 1.0
    tau_b, q_b, qd_b, per_element_b = run(tau1)
    per_element_b.sum().backward()
    assert torch.equal(q_b[[0, 2]], q[[0, 2]]) and torch.equal(qd_b[[0, 2]], qd[[0, 2]])
    assert not torch.equal(q_b[1], q[1])
    assert torch.equal(tau_b.grad[[0, 2]], grad[[0, 2]]) and not torch.equal(tau_b.grad[1], grad[1])


def test_perturb_changes_dynamics_and_keeps_state():
    B = 2
    batch = nominal_batch(B)
    batch.reset(np.tile(HOME, (B, 1)), np.zeros((B, 7)))
    q, qd = batch.state()
    tau = torch.zeros(B, 7)
    with torch.no_grad():
        q1, qd1, *_ = batch.step(q, qd, tau)
    batch.perturb([DynamicsPerturbation(), DynamicsPerturbation(payload_mass=2.0)])
    assert torch.equal(batch.state()[0], q1) and torch.equal(batch.state()[1], qd1)
    with torch.no_grad():
        q2, qd2, *_ = batch.step(q1, qd1, tau)
    assert not torch.allclose(qd2[0], qd2[1])  # the payload changes the free-fall acceleration
    with pytest.raises(ValueError):
        MjBatch([DynamicsPerturbation(actuation_delay=0.002)])


def test_torque_clamp():
    B = 1
    batch = nominal_batch(B)
    batch.reset(HOME[None], np.zeros((B, 7)))
    q, qd = batch.state()
    tau = torch.tensor([[200.0, -200.0, 1.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    q1, qd1, touch, _ = batch.step(q, qd, tau)
    assert np.array_equal(batch.sims[0].data.ctrl[:2], [TORQUE_LIMITS[0], -TORQUE_LIMITS[1]])
    (qd1**2).sum().backward()
    assert torch.equal(tau.grad[0, :2], torch.zeros(2))
    assert tau.grad[0, 2] != 0


def test_local_touch_gradient_drops_state_path():
    """With touch_grad_local the touch-sensor gradient reaches the torque of the same step but not
    the state of earlier steps (used for the deep-dive 'forcelocal' arm)."""
    import numpy as np
    import torch

    from llc.learning.mjgrad import MjBatch
    from llc.sim.nominal import NominalModel
    from llc.sim.panda import DynamicsPerturbation, HOME

    nominal = NominalModel()
    q_start, _ = nominal.ik(np.array([0.45, 0.0, 0.172]), HOME, tool_axis_world=np.array([0.0, 0.0, -1.0]))
    for local in (False, True):
        batch = MjBatch([DynamicsPerturbation()], seed=0)
        batch.touch_grad_local = local
        batch.reset(q_start[None], np.zeros((1, 7)))
        q, qd = batch.state()
        q0 = q.clone().requires_grad_(True)
        qd0 = qd.clone().requires_grad_(True)
        g = torch.tensor(nominal.gravity(q_start), dtype=torch.float32)[None]
        tau = (g - 30.0 * torch.tensor(nominal.tool_jacobian(q_start)[0][2], dtype=torch.float32)[None]).requires_grad_(True)
        qq, qqd = q0, qd0
        touches = []
        for _ in range(30):
            qq, qqd, touch, _ = batch.step(qq, qqd, tau, None)
            touches.append(touch)
        loss = torch.stack(touches).square().sum()
        loss.backward()
        assert tau.grad is not None and tau.grad.abs().sum() > 0
        state_grad = q0.grad.abs().sum() + qd0.grad.abs().sum()
        if local:
            # only the state path through A remains; the touch sensor itself contributes nothing through C
            assert state_grad < 1e-3 * tau.grad.abs().sum() or state_grad == 0.0
        else:
            assert state_grad > 0.0
