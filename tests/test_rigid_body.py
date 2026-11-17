"""RigidBodyModel and penalty contact model against MuJoCo on the Panda torque scene (free space)."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from llc.dynamics.contact import ContactParams, SceneGeometry, contact_force
from llc.dynamics.rigid_body import RigidBodyModel

SCENE = Path(__file__).resolve().parents[1] / "llc" / "sim" / "assets" / "panda_scene.xml"
N_STATES = 20
TOL = dict(rtol=1e-6, atol=1e-9)


def t(x, dtype=torch.float64):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def name2id(model, kind, name):
    return mujoco.mj_name2id(model, getattr(mujoco.mjtObj, f"mjOBJ_{kind}"), name)


@pytest.fixture(scope="module")
def mj():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    return model, mujoco.MjData(model)


@pytest.fixture(scope="module")
def rbm(mj):
    return RigidBodyModel(mj[0])


@pytest.fixture(scope="module")
def states(mj):
    """Random free-space states: q in the joint ranges with no contacts and no active constraints
    (``ncon == nefc == 0``), qd in [-2, 2], tau within the actuator ctrlrange."""
    model, data = mj
    rng = np.random.default_rng(0)
    lo, hi = model.jnt_range.T
    ctrl_lo, ctrl_hi = model.actuator_ctrlrange.T
    q, qd, tau = [], [], []
    while len(q) < N_STATES:
        cand = rng.uniform(lo, hi)
        data.qpos[:], data.qvel[:] = cand, 0
        mujoco.mj_forward(model, data)
        if data.ncon == 0 and data.nefc == 0:
            q.append(cand)
            qd.append(rng.uniform(-2, 2, model.nv))
            tau.append(rng.uniform(ctrl_lo, ctrl_hi))
    return np.array(q), np.array(qd), np.array(tau)


def test_kinematics_vs_mujoco(mj, rbm, states):
    model, data = mj
    q = states[0]
    sid = name2id(model, "SITE", "tool_tip")
    xpos, xmat, site, jac = [], [], [], []
    for qi in q:
        data.qpos[:] = qi
        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, sid)
        xpos.append(data.xpos.copy())
        xmat.append(data.xmat.reshape(-1, 3, 3).copy())
        site.append(data.site_xpos[sid].copy())
        jac.append(jacp)
    fk = rbm.fk(t(q))
    assert torch.allclose(fk["pos"], t(xpos), **TOL)
    assert torch.allclose(fk["rot"], t(xmat), **TOL)
    assert torch.allclose(fk["tool_pos"], t(site), **TOL)
    assert torch.allclose(rbm.tool_pos(t(q)), t(site), **TOL)
    assert torch.allclose(rbm.tool_jacobian(t(q)), t(jac), **TOL)


def test_dynamics_vs_mujoco(mj, rbm, states):
    model, data = mj
    q, qd, tau = states
    rng = np.random.default_rng(1)
    qdd = rng.uniform(-5, 5, q.shape)
    f_tool = rng.uniform(-20, 20, (len(q), 3))
    sid, tool = name2id(model, "SITE", "tool_tip"), name2id(model, "BODY", "tool")
    M, bias, qacc, qacc_f, qfrc_inv = [], [], [], [], []
    for i in range(len(q)):
        data.qpos[:], data.qvel[:], data.ctrl[:], data.qfrc_applied[:] = q[i], qd[i], tau[i], 0
        mujoco.mj_forward(model, data)
        assert data.ncon == 0 and data.nefc == 0
        M.append(np.zeros((model.nv, model.nv)))
        mujoco.mj_fullM(model, data, M[-1])
        bias.append(data.qfrc_bias.copy())
        qacc.append(data.qacc.copy())
        # World-frame force at the site (xfrc_applied would act at the body COM, so use mj_applyFT).
        mujoco.mj_applyFT(model, data, f_tool[i], np.zeros(3), data.site_xpos[sid], tool, data.qfrc_applied)
        mujoco.mj_forward(model, data)
        qacc_f.append(data.qacc.copy())
        data.qfrc_applied[:] = 0
        data.qacc[:] = qdd[i]
        mujoco.mj_inverse(model, data)
        qfrc_inv.append(data.qfrc_inverse.copy())
    q, qd, tau, qdd, f_tool = map(t, (q, qd, tau, qdd, f_tool))
    assert torch.allclose(rbm.mass_matrix(q), t(M), **TOL)
    assert torch.allclose(rbm.bias(q, qd), t(bias), **TOL)
    # qfrc_inverse = M qdd + bias - qfrc_passive, i.e. it includes the damping torque.
    assert torch.allclose(rbm.inverse_dynamics(q, qd, qdd) - rbm.passive(qd), t(qfrc_inv), **TOL)
    assert torch.allclose(rbm.forward_dynamics(q, qd, tau), t(qacc), **TOL)
    assert torch.allclose(rbm.forward_dynamics(q, qd, tau, f_tool), t(qacc_f), **TOL)


def test_gradcheck_forward_dynamics(mj, states):
    rbm = RigidBodyModel(mj[0])
    q, qd, tau = (t(x[:2]).requires_grad_() for x in states)
    params = (rbm.mass, rbm.com, rbm.inertia)
    for p in params:
        p.requires_grad_()

    def fn(q, qd, tau, mass, com, inertia):
        rbm.mass, rbm.com, rbm.inertia = mass, com, inertia
        return rbm.forward_dynamics(q, qd, tau)

    assert torch.autograd.gradcheck(fn, (q, qd, tau, *params))


def test_batch_consistency(rbm, states):
    q, qd, tau = map(t, states)
    batched = rbm.forward_dynamics(q, qd, tau), rbm.tool_jacobian(q), rbm.mass_matrix(q)
    for i in range(len(q)):
        single = (
            rbm.forward_dynamics(q[i : i + 1], qd[i : i + 1], tau[i : i + 1]),
            rbm.tool_jacobian(q[i : i + 1]),
            rbm.mass_matrix(q[i : i + 1]),
        )
        for a, b in zip(batched, single):
            assert torch.allclose(a[i], b[0], rtol=1e-12, atol=1e-12)


def test_rollout_vs_mujoco(mj, rbm):
    """200 steps of PD + gravity compensation, each simulator in closed loop on its own state.  MuJoCo
    integrates with implicitfast; step() treats damping implicitly in the same way.  The start is the
    home keyframe with joint 2 tilted back by 0.3 rad: in the home pose itself the arm intersects the
    wall, and the rollout must stay in free space (asserted every step)."""
    model, data = mj
    mujoco.mj_resetDataKeyframe(model, data, name2id(model, "KEY", "home"))
    data.qpos[1] -= 0.3
    dt = model.opt.timestep
    ctrl_lo, ctrl_hi = (t(x) for x in model.actuator_ctrlrange.T)
    q_ref = t(data.qpos.copy()) + 0.2
    q, qd = t(data.qpos.copy())[None], torch.zeros(1, model.nv, dtype=torch.float64)

    def controller(q, qd):
        tau = 50 * (q_ref - q) - 5 * qd + rbm.bias(q, torch.zeros_like(qd))
        return torch.clamp(tau, ctrl_lo, ctrl_hi)

    for _ in range(200):
        data.ctrl[:] = controller(t(data.qpos)[None], t(data.qvel)[None])[0].numpy()
        mujoco.mj_step(model, data)
        assert data.ncon == 0 and data.nefc == 0
        q, qd = rbm.step(q, qd, controller(q, qd), dt)
    deviation = np.abs(q[0].numpy() - data.qpos).max()
    print(f"\nrollout: max |q_torch - q_mujoco| after 200 steps = {deviation:.3e} rad")
    assert deviation < 2e-3


def test_float32_matches_float64(mj, rbm, states):
    rbm32 = RigidBodyModel(mj[0], dtype=torch.float32)
    q, qd, tau = map(t, states)
    qacc64 = rbm.forward_dynamics(q, qd, tau)
    qacc32 = rbm32.forward_dynamics(q.float(), qd.float(), tau.float()).double()
    assert (qacc32 - qacc64).abs().max() < 1e-3 * qacc64.abs().max()


# ------------------------------------------------------------------------------------- contact
@pytest.fixture(scope="module")
def geometry(mj):
    return SceneGeometry.from_model(mj[0])


def test_contact_geometry_from_model(mj, geometry):
    model = mj[0]
    r = model.geom_size[name2id(model, "GEOM", "tool_tip")][0]
    table_pos, table_size = (model.geom_pos[name2id(model, "GEOM", "table")], model.geom_size[name2id(model, "GEOM", "table")])
    wall_pos, wall_size = (model.geom_pos[name2id(model, "GEOM", "wall")], model.geom_size[name2id(model, "GEOM", "wall")])
    assert geometry.radius == r
    assert geometry.faces[1].offset == pytest.approx(table_pos[2] + table_size[2])
    assert -geometry.faces[2].offset == pytest.approx(wall_pos[0] - wall_size[0])


def test_contact_force(geometry):
    params = ContactParams()
    r = geometry.radius
    z_table, x_wall = geometry.faces[1].offset, -geometry.faces[2].offset
    x_table = 0.5 * (geometry.faces[1].bounds[0][1] + geometry.faces[1].bounds[0][2])
    free = t([[0.3, 0.0, 0.5]])
    f, d = contact_force(free, t([[1.0, 1.0, 1.0]]), params, geometry)
    assert torch.all(f == 0) and torch.all(d == 0)

    # Floor: pressed in by 5 mm while sliding in +x and moving down.
    f, d = contact_force(t([[0.1, 0.1, r - 0.005]]), t([[0.1, 0.0, -0.1]]), params, geometry)
    assert d[0, 0] > 0 and d[0, 1] == 0 and d[0, 2] == 0
    assert f[0, 2] > 0 and f[0, 0] < 0 and f[0, 1] == 0
    # Table top: pressed in with no velocity -> purely normal force k_n * d.
    f, d = contact_force(t([[x_table, 0.0, z_table + r - 0.003]]), torch.zeros(1, 3, dtype=torch.float64), params, geometry)
    assert torch.allclose(f[0], t([0.0, 0.0, params.k_n * 0.003]))
    # Same height but far outside the table's y extent (and the wall's) -> no force.
    y_out = geometry.faces[1].bounds[1][2] + 1.0
    f, _ = contact_force(t([[x_table, y_out, z_table + r - 0.003]]), torch.zeros(1, 3, dtype=torch.float64), params, geometry)
    assert torch.all(f == 0)
    # Wall front face: force along -x.
    f, d = contact_force(t([[x_wall - r + 0.004, 0.0, 0.45]]), t([[0.05, 0.1, 0.0]]), params, geometry)
    assert d[0, 2] > 0 and f[0, 0] < 0 and f[0, 1] < 0


def test_contact_gradcheck(geometry):
    params = ContactParams()
    r = geometry.radius
    z_table, x_wall = geometry.faces[1].offset, -geometry.faces[2].offset
    x_table = 0.5 * (geometry.faces[1].bounds[0][1] + geometry.faces[1].bounds[0][2])
    pos = t([[0.1, 0.1, r - 0.005], [x_table, 0.05, z_table + r - 0.003], [x_wall - r + 0.004, 0.0, 0.45]]).requires_grad_()
    vel = t([[0.1, -0.05, -0.1], [0.2, 0.1, 0.05], [0.05, 0.1, -0.02]]).requires_grad_()
    assert torch.autograd.gradcheck(lambda p, v: contact_force(p, v, params, geometry)[0], (pos, vel))
