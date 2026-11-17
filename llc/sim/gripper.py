"""Plant v3 (ADR 0009): the Panda with a Robotiq 2F-85 gripper, as on the DROID robot, with the
gripper baked at a fixed opening into one rigid tool body.

The gripper is not part of the controlled system (advisor, 2026-11-16): its four-bar fingers are
settled at a commanded opening in a standalone simulation of the Menagerie model, every gripper body
gets that settled pose, the joints, tendon, equality constraints and actuator are removed, and the
jointless tree is attached to the Panda flange and fused into a single body ``tool`` by the compiler
(``fusestatic``). The tool point ``tool_tip`` is the centre of the two fingertip end faces, the
``tool_touch`` box site around it feeds the touch sensor, and the pad boxes are the tool geoms. The
result is written as a plain MJCF next to the sphere-tool scene and loaded like it.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets"
GRIPPER_XML = ASSETS / "robotiq_2f85" / "2f85.xml"
BASE_SCENE = ASSETS / "panda_scene.xml"
HAND_POS = (0.0, 0.0, 0.107)  # Menagerie: the hand mounts 107 mm along the flange axis, rotated -45 degrees about it
HAND_QUAT = (0.9238795, 0.0, 0.0, -0.3826834)
PAD_TIP_HALF = 0.009375  # half-length of the outer pad box along the finger (pad_box1)
DRIVER_CLOSED = 0.8  # rad, the driver joint range end (fingers closed)
GRIPPER_PREFIX = "g_"  # attached gripper element names: g_right_pad1 ...
PAD_NAMES = ("right_pad1", "right_pad2", "left_pad1", "left_pad2")
PAD_GEOMS = tuple(f"{GRIPPER_PREFIX}{n}" for n in PAD_NAMES)
# Pad contact parameters: the scene's explicit spring-damper reference with the graded impedance (as the sphere tool of ADR 0001), so
# that tool-environment contacts on plant v3 have the same model as on v2; the Menagerie pads (solref 0.004 1, solimp 0.95 0.99, priority 1)
# gave 400 to 1600 N impacts at the protocol's approach speeds. Friction as the environment.
PAD_SOLREF = (-20000.0, -200.0)
PAD_SOLIMP = (0.3, 0.95, 0.005, 0.5, 2.0)
PAD_FRICTION = (0.5, 0.005, 0.0001)


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _quat_inv(q):
    out = np.zeros(4)
    mujoco.mju_negQuat(out, q)
    return out


def _settle(driver_angle: float, steps: int = 4000):
    """Standalone Menagerie gripper settled at ``driver_angle`` (0 = open, 0.8 = closed), no gravity."""
    model = mujoco.MjModel.from_xml_path(str(GRIPPER_XML))
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    data.ctrl[0] = driver_angle * 255.0 / DRIVER_CLOSED
    for _ in range(steps):
        mujoco.mj_step(model, data)
    assert np.abs(data.qvel).max() < 1e-3, "the fingers did not settle"
    return model, data


def _rel_pose(data, b: int, root: int) -> tuple[np.ndarray, np.ndarray]:
    """Pose of body ``b`` in the frame of body ``root`` (world data after mj_forward)."""
    r_root = data.xmat[root].reshape(3, 3)
    return r_root.T @ (data.xpos[b] - data.xpos[root]), _quat_mul(_quat_inv(data.xquat[root]), data.xquat[b])


def _merged_inertia(model, data, root: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Mass, centre of mass, inertia eigenvalues and inertial-frame quaternion of the subtree of ``root``,
    expressed in the frame of ``root`` (parallel-axis merge of the bodies' own inertias)."""
    r_root = data.xmat[root].reshape(3, 3)
    bodies = [b for b in range(model.nbody) if _in_subtree(model, b, root)]
    mass = sum(float(model.body_mass[b]) for b in bodies)
    com_w = sum(float(model.body_mass[b]) * data.xipos[b] for b in bodies) / mass
    inertia_w = np.zeros((3, 3))
    for b in bodies:
        r_i = data.ximat[b].reshape(3, 3)
        inertia_b = r_i @ np.diag(model.body_inertia[b]) @ r_i.T
        d = data.xipos[b] - com_w
        inertia_w += inertia_b + float(model.body_mass[b]) * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    inertia_root = r_root.T @ inertia_w @ r_root
    w, v = np.linalg.eigh(inertia_root)
    if np.linalg.det(v) < 0:
        v[:, 0] *= -1
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, v.reshape(9))
    return mass, r_root.T @ (com_w - data.xpos[root]), w, quat


def _in_subtree(model, b: int, root: int) -> bool:
    while b != 0:
        if b == root:
            return True
        b = int(model.body_parentid[b])
    return False


def bake_gripper(driver_angle: float) -> tuple[mujoco.MjSpec, float, float]:
    """The Menagerie gripper as a single jointless body (its root ``base_mount``) holding every geom at the
    settled pose for ``driver_angle`` and the merged inertia. Returns the spec, the settled driver angle
    and the subtree mass."""
    model, data = _settle(driver_angle)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_mount")
    g = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    for coll in (g.joints, g.equalities, g.tendons, g.actuators, g.sites):
        for e in list(coll):
            g.delete(e)
    base = g.body("base_mount")
    moved = []
    for body in list(g.bodies):
        if body.name in ("world", "base_mount"):
            continue
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.name)
        pos_b, quat_b = _rel_pose(data, b, root)
        r_b = np.zeros(9)
        mujoco.mju_quat2Mat(r_b, quat_b)
        r_b = r_b.reshape(3, 3)
        for geom in body.geoms:  # the geom's own (user) frame composed with the settled body pose
            new = base.add_geom(default=geom.classname)
            new.type, new.meshname, new.material = geom.type, geom.meshname, geom.material
            new.pos = pos_b + r_b @ np.asarray(geom.pos)
            new.quat = _quat_mul(quat_b, np.asarray(geom.quat))
            moved.append((new, geom.name))
    for child in list(base.bodies):  # deleting a direct child removes its subtree
        g.delete(child)
    for new, name in moved:  # names after the originals are gone
        new.name = name
        if name in PAD_NAMES:  # the pads contact the environment like the sphere tool did: the scene's graded soft contact, no priority
            new.solref = np.array(PAD_SOLREF)
            new.solimp = np.array(PAD_SOLIMP)
            new.friction = np.array(PAD_FRICTION)
            new.priority = 0
    mass, com, inertia, iquat = _merged_inertia(model, data, root)
    base.explicitinertial = True
    base.mass, base.ipos, base.inertia, base.iquat = mass, com, inertia, iquat
    return g, float(data.qpos[0]), mass


def build_scene(driver_angle: float = DRIVER_CLOSED, out: Path | None = None) -> Path:
    """Write the plant v3 scene for a fixed gripper opening and return its path."""
    out = out or ASSETS / f"panda_2f85_{int(round(1000 * driver_angle)):04d}.xml"
    g, q_driver, mass = bake_gripper(driver_angle)
    for m in g.meshes:  # absolute mesh paths for the compile; the written scene gets them relative to the assets directory
        m.file = str(ASSETS / "robotiq_2f85" / "assets" / Path(m.file).name)
        m.scale = np.array([0.001, 0.001, 0.001])
    g.compiler.meshdir = ""
    p = mujoco.MjSpec.from_file(str(BASE_SCENE))
    p.compiler.meshdir = ""
    for m in p.meshes:
        m.file = str(ASSETS / "panda" / Path(m.file).name)
    for s in list(p.sensors):
        p.delete(s)
    p.delete(p.body("tool"))
    frame = p.body("link7").add_frame(pos=np.array(HAND_POS), quat=np.array(HAND_QUAT))
    tool = frame.attach_body(g.body("base_mount"), GRIPPER_PREFIX, "")
    tool.name = "tool"
    model = p.compile()
    # tool point: centre of the two outer pad end faces, in the tool body's frame
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")
    tips = []
    for name in (PAD_GEOMS[0], PAD_GEOMS[2]):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_bodyid[gid] == bid, f"{name} is not on the tool body"
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, model.geom_quat[gid])
        tips.append(model.geom_pos[gid] + r.reshape(3, 3) @ np.array([0.0, 0.0, PAD_TIP_HALF]))
    tcp = np.mean(tips, axis=0)
    gap = float(np.linalg.norm(tips[0] - tips[1]))
    tool_spec = p.body("tool")
    tip = tool_spec.add_site(name="tool_tip", pos=tcp, size=np.array([0.005, 0.005, 0.005]))
    tip.type = mujoco.mjtGeom.mjGEOM_SPHERE
    touch = tool_spec.add_site(name="tool_touch", pos=tcp, size=np.array([0.02, 0.02, 0.01]), rgba=np.array([0.0, 0.0, 0.0, 0.0]))
    touch.type = mujoco.mjtGeom.mjGEOM_BOX
    p.add_sensor(name="tool_touch", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="tool_touch")
    model = p.compile()
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    assert model.nv == 7 and model.nu == 7 and names == ["world", "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "tool"], (model.nv, model.nu, names)
    import os

    rel = os.path.relpath(ASSETS, out.parent)
    xml = p.to_xml().replace(str(ASSETS) + "/", "" if rel == "." else rel + "/")  # mesh paths relative to the written scene
    xml = xml.replace("with a fixed spherical tool", "with the Robotiq 2F-85 baked into the rigid body tool (see the header)")
    header = (f"<!-- Generated by llc/sim/gripper.py (plant v3, ADR 0009): Panda with the Robotiq 2F-85 baked at driver angle {driver_angle:.3f} rad\n"
              f"     (settled {q_driver:.3f} rad; fingertip gap {1e3 * gap:.1f} mm; gripper mass {mass:.3f} kg) into the rigid body 'tool'; tool point at the fingertip end faces. Do not edit. -->\n")
    out.write_text(xml.replace("<mujoco model=", header + "<mujoco model=", 1))
    return out


if __name__ == "__main__":
    import sys

    angle = float(sys.argv[1]) if len(sys.argv) > 1 else DRIVER_CLOSED
    path = build_scene(angle)
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    d.qpos[:] = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tool")
    print("wrote", path, "| nbody", m.nbody, "nv", m.nv, "ngeom", m.ngeom)
    print("bodies:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)])
    print("tool body id", bid, "mass", m.body_mass[bid], "subtree", m.body_subtreemass[bid], "ipos", m.body_ipos[bid])
    print("tool geoms:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom) if m.geom_bodyid[i] == bid])
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
    print("tool_tip at home:", d.site_xpos[sid], "tool z axis:", d.xmat[bid].reshape(3, 3)[:, 2], "sensor:", m.nsensor)
