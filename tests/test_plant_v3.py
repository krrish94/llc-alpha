"""Plant v3 (ADR 0009): the gripper scene, the tool point, the protocol v3 metrics and guards.
These tests run on whatever plant LLC_PLANT selects for the scene build, but build the v3 scene
explicitly and skip the plant-dependent checks unless LLC_PLANT=v3."""
import os

import mujoco
import numpy as np
import pytest

from llc.sim.gripper import DRIVER_CLOSED, PAD_GEOMS, build_scene

V3 = os.environ.get("LLC_PLANT") == "v3"


def test_gripper_scene_builds_as_one_rigid_tool_with_seven_dof(tmp_path):
    path = build_scene(DRIVER_CLOSED, tmp_path / "scene.xml")
    m = mujoco.MjModel.from_xml_path(str(path))
    assert m.nv == 7 and m.nu == 7
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tool")
    assert 0.9 < m.body_mass[bid] < 1.3  # the 2F-85 (about 0.9 kg) plus its mount
    for name in PAD_GEOMS:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert m.geom_bodyid[gid] == bid and m.geom_priority[gid] == 0
    d = mujoco.MjData(m)
    d.qpos[:] = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
    z_axis = d.xmat[bid].reshape(3, 3)[:, 2]
    assert np.allclose(z_axis, [0, 0, -1], atol=1e-3)  # the tool axis points down at the home pose, as the sphere tool did
    assert abs(d.site_xpos[sid][2] - 0.451) < 0.02  # tool point height at home, about where the sphere tip was
    assert m.nsensor == 1


@pytest.mark.skipif(not V3, reason="LLC_PLANT=v3 selects the gripper plant")
def test_v3_contact_case_presses_with_the_pads_and_reports_release_metrics():
    import json

    from llc.controllers.factory import build_controller
    from llc.eval.cases_v2 import make_case_v2
    from llc.eval.metrics import compute_metrics
    from llc.eval.rollout import rollout
    from llc.sim.nominal import NominalModel
    from llc.sim.panda import PandaSim

    nominal = NominalModel()
    spec = json.load(open("runs/d1/gains_shared.json"))["spec"]
    case = make_case_v2("contact2", "val", 0, nominal)
    assert case.meta["protocol"] == "v3" and 0.004 <= case.meta["depth"] <= 0.014
    ep = rollout(PandaSim(case.perturbation, seed=case.seed), build_controller(spec, nominal, case), case)
    m = compute_metrics(ep, case)
    assert (ep.n_tool_contacts > 0).mean() > 0.3 and m["other_contact"] < 0.05
    case_b = make_case_v2("blocked", "val", 0, nominal)
    m_b = compute_metrics(rollout(PandaSim(case_b.perturbation, seed=case_b.seed), build_controller(spec, nominal, case_b), case_b), case_b)
    assert np.isfinite(m_b["release_reaction"]) and np.isfinite(m_b["release_settle"]) and np.isfinite(m_b["composite_cap"])


@pytest.mark.skipif(not V3, reason="LLC_PLANT=v3 selects the gripper plant")
def test_v3_link_clearance_is_signed_and_flags_a_pose_in_the_table():
    from llc.sim.nominal import NominalModel
    from llc.sim.panda import HOME

    nominal = NominalModel()
    assert nominal.link_clearance(HOME, "table") >= 0.05  # far away: capped at distmax
    q, err = nominal.ik(np.array([0.45, 0.0, 0.12]), HOME)  # tool point 3 cm below the table top: the fingers are inside
    assert err < 1e-2 and nominal.link_clearance(q, "table") < 0.0


@pytest.mark.skipif(not V3, reason="LLC_PLANT=v3 selects the gripper plant")
def test_v3_task_families_run_and_report_their_metrics():
    import json

    from llc.controllers.factory import build_controller
    from llc.eval.cases_v2 import make_case_v2
    from llc.eval.metrics import compute_metrics
    from llc.eval.rollout import rollout
    from llc.sim.nominal import NominalModel
    from llc.sim.panda import PandaSim

    nominal = NominalModel()
    spec = json.load(open("runs/d1/gains_shared.json"))["spec"]
    for regime, keys in (("push", ("push_error", "push_lateral", "push_yaw", "object_force_peak_20ms", "task_success")), ("pick", ("grasp_success", "pregrasp_disturbance", "place_error", "place_tilt", "place_force_peak_20ms", "lift_rmse", "task_success"))):
        case = make_case_v2(regime, "test", 0, nominal)
        task = case.meta["task"]
        ep = rollout(PandaSim(case.perturbation, seed=case.seed, scene=task["scene"]), build_controller(spec, nominal, case), case)
        m = compute_metrics(ep, case)
        assert ep.obj_pos is not None and all(np.isfinite(m[k]) for k in keys) and np.isfinite(m["composite_cap"])
        if regime == "pick":
            assert abs(ep.obj_pos[-1][2] - 0.17) < 0.01  # the block ends on the table top (0.15 m) with its 2 cm half-height
            assert m["place_tilt"] < np.radians(15)
