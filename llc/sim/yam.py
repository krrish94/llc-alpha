"""Plant ``yam`` (ADR 0011, decision 8): the i2rt YAM (MuJoCo Menagerie model, MIT licence) with its
two-finger gripper baked at a fixed opening into the rigid body ``tool``, torque actuators on the six arm
joints (the servo layer of ``llc.sim.servo`` sits in our code), and the protocol's environment (floor,
table block, wall) scaled to the arm's reach. Built with MjSpec and written next to the Panda scenes.

The tool point ``tool_tip`` is the centre of the two pad inner faces at the fingertip end; the pad boxes
and the fingertip spheres are the tool geoms with the contact model of ADR 0001; every other finger geom
counts as a robot link. The gripper's opening is the finger slide (0 = closed, 0.0226 = a 46 mm gap).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets"
YAM_XML = ASSETS / "i2rt_yam" / "yam.xml"
PAD_NAMES = ("lf_pad", "rf_pad")
TIP_NAMES = tuple(f"{side}_tip{i}" for side in ("lf", "rf") for i in range(6))
TOOL_GEOMS = PAD_NAMES + TIP_NAMES
PAD_SOLREF, PAD_SOLIMP, PAD_FRICTION = (-20000.0, -200.0), (0.3, 0.95, 0.005, 0.5, 2.0), (0.5, 0.005, 0.0001)
FINGER_BODIES = ("link_left_finger", "lf_rot", "lf_down", "link_right_finger", "rf_rot", "rf_down")
ENV = {"table_pos": (0.34, 0.0, 0.03), "table_size": (0.13, 0.27, 0.03), "wall_pos": (0.60, 0.0, 0.30), "wall_size": (0.02, 0.27, 0.30)}


def _settled(opening: float):
    """The Menagerie model with the fingers at ``opening`` (slide displacement per finger), forward kinematics done."""
    model = mujoco.MjModel.from_xml_path(str(YAM_XML))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[6], data.qpos[7] = opening, -opening
    mujoco.mj_forward(model, data)
    return model, data


def _rel(data, b: int, root: int):
    r_root = data.xmat[root].reshape(3, 3)
    q_inv, q = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(q_inv, data.xquat[root]); mujoco.mju_mulQuat(q, q_inv, data.xquat[b])
    return r_root.T @ (data.xpos[b] - data.xpos[root]), q


def _merged_inertia(model, data, bodies: list[int], root: int):
    r_root = data.xmat[root].reshape(3, 3)
    mass = sum(float(model.body_mass[b]) for b in bodies)
    com_w = sum(float(model.body_mass[b]) * data.xipos[b] for b in bodies) / mass
    inertia_w = np.zeros((3, 3))
    for b in bodies:
        r_i = data.ximat[b].reshape(3, 3)
        d = data.xipos[b] - com_w
        inertia_w += r_i @ np.diag(model.body_inertia[b]) @ r_i.T + float(model.body_mass[b]) * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    inertia_root = r_root.T @ inertia_w @ r_root
    w, v = np.linalg.eigh(inertia_root)
    if np.linalg.det(v) < 0:
        v[:, 0] *= -1
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, v.reshape(9))
    return mass, r_root.T @ (com_w - data.xpos[root]), w, quat


def build_scene(opening: float = 0.0, out: Path | None = None) -> Path:
    """Write the plant scene for a fixed gripper opening and return its path."""
    out = out or ASSETS / f"yam_{int(round(1000 * opening)):04d}.xml"
    model, data = _settled(opening)
    spec = mujoco.MjSpec.from_file(str(YAM_XML))
    for m in spec.meshes:
        m.file = str(ASSETS / "i2rt_yam" / "assets" / Path(m.file).name)
    spec.compiler.meshdir = ""
    # the tool body: the finger geoms at the settled pose on one jointless body under link_6, with the merged inertia
    link6 = spec.body("link_6")
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
    finger_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in FINGER_BODIES]
    tool = link6.add_body(name="tool", pos=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    moved = []
    pad_count = {"lf": 0, "rf": 0}
    for name, b in zip(FINGER_BODIES, finger_ids):
        body = spec.body(name)
        pos_b, quat_b = _rel(data, b, root)
        r_b = np.zeros(9); mujoco.mju_quat2Mat(r_b, quat_b); r_b = r_b.reshape(3, 3)
        side = "lf" if name.startswith(("link_left", "lf")) else "rf"
        for geom in list(body.geoms):
            new = tool.add_geom(default=geom.classname)
            new.type, new.meshname, new.material, new.size, new.rgba = geom.type, geom.meshname, geom.material, np.array(geom.size), np.array(geom.rgba)
            new.pos = pos_b + r_b @ np.asarray(geom.pos)
            new.quat = _mul(quat_b, np.asarray(geom.quat))
            kind = None
            if name.endswith("_down") and geom.type == mujoco.mjtGeom.mjGEOM_BOX and abs(float(geom.size[2]) - 0.04) < 1e-6:
                kind = f"{side}_pad"
            elif geom.classname.name == "sphere_collision":
                kind = f"{side}_tip{pad_count[side]}"; pad_count[side] += 1
            moved.append((new, kind))
    for name in FINGER_BODIES:
        if spec.body(name) is not None and spec.body(name).parent is link6:
            spec.delete(spec.body(name))  # deleting a direct child removes its subtree
    for coll in (spec.equalities, spec.actuators):
        for e in list(coll):
            if "finger" in (e.name or "") or e.name == "gripper" or getattr(e, "joint1", "") in ("left_finger", "right_finger") or getattr(e, "target", "") in ("left_finger", "right_finger"):
                spec.delete(e)
    for new, kind in moved:
        if kind is not None:
            new.name = kind
            new.solref, new.solimp, new.friction, new.priority = np.array(PAD_SOLREF), np.array(PAD_SOLIMP), np.array(PAD_FRICTION), 0
    mass, com, inertia, iquat = _merged_inertia(model, data, finger_ids, root)
    tool.explicitinertial = True
    tool.mass, tool.ipos, tool.inertia, tool.iquat = mass, com, inertia, iquat
    # tool point: the centre of the two pad inner faces at the fingertip end, in the tool (= link_6) frame
    pads = [(n, k) for n, k in moved if k in PAD_NAMES]
    tips = []
    for g, _ in pads:
        r = np.zeros(9); mujoco.mju_quat2Mat(r, g.quat); r = r.reshape(3, 3)
        inner = -np.sign(g.pos[1]) if abs(g.pos[1]) > 1e-9 else 1.0  # the pads face each other across y
        tips.append(np.asarray(g.pos) + r @ np.array([0.0, inner * float(g.size[1]), float(g.size[2])]))
    tcp = np.mean(tips, axis=0)
    tip_site = tool.add_site(name="tool_tip", pos=tcp, size=np.array([0.005, 0.005, 0.005]))
    tip_site.type = mujoco.mjtGeom.mjGEOM_SPHERE
    touch = tool.add_site(name="tool_touch", pos=tcp, size=np.array([0.015, 0.015, 0.008]), rgba=np.array([0.0, 0.0, 0.0, 0.0]))
    touch.type = mujoco.mjtGeom.mjGEOM_BOX
    spec.add_sensor(name="tool_touch", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="tool_touch")
    # actuators: torque motors on the six arm joints (the position actuators of the Menagerie model are the servo layer's business)
    for a in list(spec.actuators):
        spec.delete(a)
    limits = (28.0, 28.0, 28.0, 10.0, 10.0, 10.0)
    for j in range(1, 7):
        act = spec.add_actuator(name=f"joint{j}", target=f"joint{j}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.gainprm[0] = 1.0; act.biasprm[:] = 0.0
        act.ctrlrange = np.array([-limits[j - 1], limits[j - 1]])
    # the environment and the cameras of the Panda scene, scaled to the reach
    wb = spec.worldbody
    wb.add_light(pos=np.array([0.0, 0.0, 2.5]), dir=np.array([0.0, 0.0, -1.0]), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_light(pos=np.array([1.5, -1.5, 2.0]), dir=np.array([-0.5, 0.5, -1.0]))
    env = {"solref": np.array([-20000.0, -200.0]), "solimp": np.array([0.3, 0.95, 0.005, 0.5, 2.0]), "friction": np.array([0.5, 0.005, 0.0001])}
    floor = wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=np.array([2.0, 2.0, 0.05]), rgba=np.array([0.96, 0.96, 0.96, 1.0]), **env)
    table = wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, pos=np.array(ENV["table_pos"]), size=np.array(ENV["table_size"]), rgba=np.array([0.88, 0.88, 0.86, 1.0]), **env)
    wall = wb.add_geom(name="wall", type=mujoco.mjtGeom.mjGEOM_BOX, pos=np.array(ENV["wall_pos"]), size=np.array(ENV["wall_size"]), rgba=np.array([0.88, 0.88, 0.86, 1.0]), **env)
    for g in (floor, table, wall):
        g.priority = 0
    wb.add_camera(name="front", pos=np.array([1.9, -1.4, 0.9]), xyaxes=np.array([0.6, 0.8, 0.0, -0.25, 0.19, 0.95]), fovy=38)
    wb.add_camera(name="side", pos=np.array([0.4, -2.0, 0.55]), xyaxes=np.array([1.0, 0.0, 0.0, 0.0, 0.2, 0.98]), fovy=36)
    wb.add_camera(name="detail", pos=np.array([-0.15, -1.1, 0.75]), xyaxes=np.array([0.880, -0.474, 0.0, 0.178, 0.331, 0.926]), fovy=32)
    for body in spec.bodies:  # the Menagerie model asks the simulator to compensate gravity on every link; the controllers do that
        body.gravcomp = 0.0
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 960, 540  # the clip renderer's frame size (as the Panda scene)
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.impratio = 10.0
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    for key in list(spec.keys):
        spec.delete(key)
    compiled = spec.compile()
    assert compiled.nv == 6 and compiled.nu == 6, (compiled.nv, compiled.nu)
    names = [mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(compiled.nbody)]
    assert names == ["world", "arm", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6", "tool"], names
    import os

    rel = os.path.relpath(ASSETS, out.parent)
    xml = spec.to_xml().replace(str(ASSETS) + "/", "" if rel == "." else rel + "/")
    header = (f"<!-- Generated by llc/sim/yam.py (plant yam, ADR 0011): the i2rt YAM (Menagerie, MIT) with its gripper baked at a finger opening of {opening:.4f} m\n"
              f"     (fingertip gap {1e3 * float(np.linalg.norm(tips[0] - tips[1])):.1f} mm; gripper mass {mass:.3f} kg) into the rigid body 'tool'; torque motors on the arm joints;\n"
              f"     tool point at the pad faces' fingertip end; floor, table block and wall as the Panda scene scaled to the reach. Do not edit. -->\n")
    out.write_text(xml.replace("<mujoco model=", header + "<mujoco model=", 1))
    return out


def _mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def task_scene(opening: float) -> Path:
    """The plant scene at a fingertip opening (m) for the task families: closed (0) for pushing, 46 mm for the pick."""
    gap_per_finger = max(0.0, (opening - 0.0008) / 2.0)
    out = ASSETS / f"yam_{int(round(1000 * gap_per_finger)):04d}.xml"
    if not out.exists():
        build_scene(gap_per_finger, out)
    return out


if __name__ == "__main__":
    import sys

    path = build_scene(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)
    m = mujoco.MjModel.from_xml_path(str(path)); d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tool")
    print("wrote", path, "| nbody", m.nbody, "nv", m.nv, "nu", m.nu, "ngeom", m.ngeom)
    print("tool geoms:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom) if m.geom_bodyid[i] == bid])
    print("tool mass", m.body_mass[bid].round(3), "subtree", m.body_subtreemass[1].round(3))
