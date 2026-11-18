"""Task families of Milestone 9 (ADR 0009): box pushing and pick-and-place on the gripper plant.

A scripted task policy emits a tool path from the randomised object pose; the reference is its
joint-space tracking target (IK as for the surface families); the object lives in a task scene
(``llc/sim/tasks.py``) that the rollout loads and resets from the case's task description. The
gripper is not controlled: pushing uses the closed gripper, pick-and-place a gripper baked at a 46 mm
opening around a 34 mm block, welded to the tool at the grasp instant if the block is inside the gap (protocol v4)
and released at the place point.
"""
from __future__ import annotations

import numpy as np

import importlib

from llc import references as R
from llc.sim.nominal import NominalModel
from llc.sim.panda import HOME, VELOCITY_LIMITS
from llc.sim.plant import PLANT
from llc.sim.tasks import BoxObject, build_task_scene

PUSH_BOX = BoxObject("box", (0.025, 0.025, 0.025), 0.2, 0.6, (0.85, 0.45, 0.2, 1.0))
PICK_BLOCK = BoxObject("block", (0.017, 0.017, 0.02), 0.3, 0.6, (0.2, 0.4, 0.85, 1.0))
PICK_OPENING = 0.046  # m, fingertip gap around the 34 mm block (6 mm of lateral tolerance per side at the grasp)
PICK_GRASP_TOL = 0.006  # m, lateral tolerance of the grasp: half of the gap minus the block's half-width
PICK_GRASP_TOL_HEIGHT = 0.009  # m, along the tool axis: the pads' half-length, so that the pads still overlap the block
PUSH_DIST = 0.10  # m
PUSH_TILT = np.radians(25.0)  # tool axis below horizontal while pushing: keeps the wrist above the table
TASK_REGIMES = ("push", "pick")


def _plant_scene(opening: float):
    """The plant's scene at a fingertip opening, from the robot's builder (``PlantSpec.task_scene_builder``)."""
    module, fn = PLANT.task_scene_builder.split(":")
    return getattr(importlib.import_module(module), fn)(opening)


def push_scene():
    return build_task_scene((PUSH_BOX,), "push", base=_plant_scene(0.0))


def pick_scene():
    return build_task_scene((PICK_BLOCK,), "pick", base=_plant_scene(PICK_OPENING), weld=PICK_BLOCK.name)


def _reference(path, times, labels, nominal, axis, rng, regime, meta, dt):
    q_start, err = R._ik_multistart(nominal, path[0], HOME, axis, rng)
    if err > 1e-3 or nominal.in_collision(q_start):
        return None
    spline = R._tool_path_to_joint(path, nominal, q_start, axis, dt)
    if spline is None:
        return None
    t = np.arange(len(path)) * dt
    q, qd, qdd = spline(t), spline(t, 1), spline(t, 2)
    if np.max(np.abs(qd) / VELOCITY_LIMITS) > 0.8:
        return None
    phases = [(times[i], times[i + 1], labels[i]) for i in range(len(labels))]
    return R.Reference(t, q, qd, qdd, regime, phases, {"tool_path": path, "surface": None, "normal": None, **meta})


def push(rng: np.random.Generator, nominal: NominalModel, speed: float, dt: float = 1e-3) -> R.Reference | None:
    """Push the box 10 cm along a random direction with the fingertip faces, the tool axis 25 degrees
    below horizontal along the push; approach from 8 cm behind the face, push at 5 cm/s, retreat 5 cm."""
    assert PLANT.tasks, "the task families live on a plant with a gripper"
    (x_lo, x_hi), (y_lo, y_hi) = PLANT.task_xy
    for _ in range(10):
        ang = rng.uniform(-np.radians(110), np.radians(110))  # push directions within 110 degrees of +x (pushing toward the base is not reachable with the tool along the push)
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
        box = np.array([rng.uniform(x_lo, x_hi - 0.02), rng.uniform(y_lo + 0.03, y_hi - 0.03), R.TABLE_TOP_Z + PUSH_BOX.size[2]])
        end = box + d * PUSH_DIST
        if x_lo - 0.02 <= end[0] <= x_hi + 0.05 and abs(end[1]) <= y_hi + 0.09 and x_lo - 0.02 <= (box - d * 0.105)[0] <= x_hi + 0.05 and abs((box - d * 0.105)[1]) <= y_hi + 0.09:
            break
    else:
        return None
    face = box - d * PUSH_BOX.size[0]
    pts = [face - d * 0.08, face, face + d * PUSH_DIST, face + d * (PUSH_DIST - 0.05)]
    speeds = [0.1 * speed, 0.05 * speed, 0.1 * speed]
    dwell = [0.3, 0.3, 0.3]
    path, times = R._tool_waypoints(pts, speeds, dwell, dt)
    labels = ["approach", "approach", "push", "push", "retreat", "retreat"]
    ref = None
    for tilt in (PUSH_TILT, PUSH_TILT + np.radians(15)):  # a steeper tool axis is tried when the shallow one is out of reach
        axis = np.array([np.cos(tilt) * d[0], np.cos(tilt) * d[1], -np.sin(tilt)])
        meta = {"task": {"name": "push", "scene": str(push_scene()), "object": PUSH_BOX.name, "object_pos": box.tolist(), "object_quat": [1.0, 0.0, 0.0, 0.0],
                         "push_dir": d.tolist(), "push_dist": PUSH_DIST, "weld": False, "tilt_deg": float(np.degrees(tilt))}, "speed": speed}
        ref = _reference(path, times, labels, nominal, axis, rng, "push", meta, dt)
        if ref is not None:
            break
    if ref is None:
        return None
    if not R._links_clear(nominal, [(ref.q[min(len(ref.q) - 1, int(times[i] / dt))], 0.0) for i in (1, 3)], pressed=None):
        return None
    return ref


def pick(rng: np.random.Generator, nominal: NominalModel, speed: float, dt: float = 1e-3) -> R.Reference | None:
    """Pick-and-place with a grasp (protocol v4, ADR 0011): approach from above the block resting on the table,
    descend over it with the gripper baked at a 46 mm opening, dwell (the weld activates 0.2 s into the dwell
    only if the block's centre is inside the fingertip gap), lift, carry to a target, descend, place (the weld is
    released 0.2 s into the dwell), lift. The block's centre is the tool point at the grasp; its yaw follows the
    tool's with a random offset within 10 degrees."""
    assert PLANT.tasks, "the task families live on a plant with a gripper"
    (x_lo, x_hi), (y_lo, y_hi) = PLANT.task_xy
    for _ in range(10):
        block = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), R.TABLE_TOP_Z + PICK_BLOCK.size[2]])
        target = block + np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 0.0])
        target[0] = np.clip(target[0], x_lo - 0.02, x_hi + 0.03); target[1] = np.clip(target[1], y_lo - 0.05, y_hi + 0.05)
        if np.hypot(*(target[:2] - block[:2])) >= 0.08:
            break
    else:
        return None
    start = block + np.array([rng.uniform(-0.10, 0.10), rng.uniform(-0.10, 0.10), 0.12])
    start[0] = np.clip(start[0], x_lo - 0.05, x_hi + 0.06)
    above_b, above_t = block + np.array([0.0, 0.0, 0.12]), target + np.array([0.0, 0.0, 0.12])
    place = target - np.array([0.0, 0.0, 0.002])
    pts = [start, above_b, block, above_b, above_t, place, above_t]
    speeds = [0.3 * speed, 0.08 * speed, 0.2 * speed, 0.3 * speed, 0.08 * speed, 0.2 * speed]
    dwell = [0.2, 0.6, 0.2, 0.2, 0.6, 0.3]
    path, times = R._tool_waypoints(pts, speeds, dwell, dt)
    labels = ["approach", "approach", "descend", "grasp", "lift", "lift", "transport", "transport", "descend", "place", "lift", "lift"]
    axis = R._table_axis(rng)
    meta = {"task": {"name": "pick", "scene": str(pick_scene()), "object": PICK_BLOCK.name, "object_pos": block.tolist(), "object_quat": [1.0, 0.0, 0.0, 0.0],
                     "target": target.tolist(), "weld": True, "weld_start": False, "grasp_time": float(times[3] + 0.2), "grasp_tol": [PICK_GRASP_TOL, PICK_GRASP_TOL_HEIGHT],
                     "release_time": float(times[9] + 0.2)}, "speed": speed}
    ref = _reference(path, times, labels, nominal, axis, rng, "pick", meta, dt)
    if ref is None:
        return None
    k_grasp, k_place = (min(len(ref.q) - 1, int(times[i] / dt)) for i in (3, 9))
    if not R._links_clear(nominal, [(ref.q[k_grasp], 0.0), (ref.q[k_place], 0.002)], pressed="table"):
        return None
    import mujoco

    _, rot = nominal.tool_frame(ref.q[k_grasp])  # the block's yaw follows the tool's heading at the grasp, within 10 degrees; upright on the table
    heading = float(np.arctan2(rot[1, 0], rot[0, 0])) + rng.uniform(-np.radians(10), np.radians(10))
    quat = np.array([np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2)])
    ref.meta["task"]["object_quat"] = quat.tolist()
    return ref


# ----------------------------------------------------------------------------- Milestone 12 families (ADR 0012, protocol v5)
from llc.sim.tasks import Fixture, JointedObject  # noqa: E402

PEG = BoxObject("peg", (0.010, 0.010, 0.025), 0.1, 0.4, (0.2, 0.2, 0.8, 1.0))  # a 20 mm square peg, 50 mm long, hanging from the fingertips
PEG_FIXTURES = {"c075": Fixture(hole_half=0.01075), "c150": Fixture(hole_half=0.0115)}  # 0.75 and 1.5 mm of clearance per side
DRAWER = JointedObject("drawer", (0.012, 0.08, 0.05), "slide", (1.0, 0.0, 0.0), (0.0, 0.10), mass=1.0, damping=20.0, frictionloss=3.0)
DOOR = JointedObject("door", (0.10, 0.01, 0.06), "hinge", (0.0, 0.0, 1.0), (0.0, np.radians(40.0)), mass=0.8, damping=0.5, frictionloss=0.3, geom_pos=(0.10, 0.0, 0.0))
DOOR_OPEN = np.radians(40.0)
TASK_REGIMES_V5 = ("push", "pick", "wipe", "erase", "peg", "drawer", "door")


def erase_scene():
    return pick_scene()  # the held block of the pick, welded from the start


def peg_scene(variant: str):
    return build_task_scene((PEG, PEG_FIXTURES[variant]), f"peg_{variant}", base=_plant_scene(0.0), weld=PEG.name, weld_offset=(0.0, 0.0, PEG.size[2]))


def drawer_scene():
    return build_task_scene((DRAWER,), "drawer", base=_plant_scene(0.0))


def door_scene():
    return build_task_scene((DOOR,), "door", base=_plant_scene(0.0))


def _tool_quat(nominal: NominalModel, q: np.ndarray) -> list[float]:
    import mujoco

    _, rot = nominal.tool_frame(q)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, float).reshape(9))
    return quat.tolist()


def _held_object_pose(nominal: NominalModel, q: np.ndarray, offset_along_axis: float) -> tuple[list[float], list[float]]:
    """The pose of an object welded to the tool at the start: the tool point plus an offset along the tool axis."""
    pos, rot = nominal.tool_frame(q)
    return (np.asarray(pos) + offset_along_axis * np.asarray(rot)[:, 2]).tolist(), _tool_quat(nominal, q)


def erase(rng: np.random.Generator, nominal: NominalModel, speed: float, dt: float = 1e-3) -> R.Reference | None:
    """Erasing with a held tool (protocol v5): the pick's block held from the start, its bottom face pressed on the table 4 to
    8 mm deep (the block's half-height below the tool point) and slid 10 cm three times (there, back, there), lifted. The
    press-and-slide metrics on the block-table force."""
    assert PLANT.tasks
    (x_lo, x_hi), (y_lo, y_hi) = PLANT.task_xy
    depth = float(rng.uniform(0.004, 0.008))
    for _ in range(10):
        p0 = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo + 0.02, y_hi - 0.02), R.TABLE_TOP_Z + PICK_BLOCK.size[2] - depth])
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
        p1 = p0 + 0.10 * d
        if x_lo - 0.02 <= p1[0] <= x_hi + 0.04 and y_lo - 0.04 <= p1[1] <= y_hi + 0.04:
            break
    else:
        return None
    above = p0 + np.array([0.0, 0.0, 0.05])
    pts = [above, p0, p1, p0, p1, above]
    speeds = [0.1 * speed, 0.1 * speed, 0.1 * speed, 0.1 * speed, 0.15 * speed]
    dwell = [0.3, 0.1, 0.1, 0.1, 0.3]
    path, times = R._tool_waypoints(pts, speeds, dwell, dt)
    labels = ["approach", "press", "slide", "press", "slide", "press", "slide", "press", "release", "free"]  # the descent counts as approach; the dwells press, the passes slide
    axis = R._table_axis(rng)
    meta = {"task": {"name": "erase", "scene": str(erase_scene()), "object": PICK_BLOCK.name, "weld": True, "weld_start": True, "release_time": 1e9, "depth": depth},
            "speed": speed, "depth": depth, "normal": [0.0, 0.0, 1.0], "surface": "table"}
    ref = _reference(path, times, labels, nominal, axis, rng, "erase", meta, dt)
    if ref is None:
        return None
    ref.meta["surface"], ref.meta["normal"] = None, [0.0, 0.0, 1.0]
    ref.meta["oracle_plane"] = [0.0, 0.0, R.TABLE_TOP_Z + PICK_BLOCK.size[2]]  # the tool point's plane when the held block's bottom face touches the table
    ref.meta["oracle_contact_eps"] = 0.01
    ref.meta["task"]["object_pos"], ref.meta["task"]["object_quat"] = _held_object_pose(nominal, ref.q[0], 0.0)
    k_press = min(len(ref.q) - 1, int(times[1] / dt))
    if not R._links_clear(nominal, [(ref.q[k_press], depth)], pressed="table"):
        return None
    return ref


def peg(rng: np.random.Generator, nominal: NominalModel, speed: float, dt: float = 1e-3) -> R.Reference | None:
    """Peg-in-hole with a chamfer (protocol v5, BPA-like): the square peg held from the start hangs 5 cm below the fingertips;
    the fixture (0.75 or 1.5 mm of clearance per side, alternating) sits on the table; the reference aims 2 to 6 mm beside the
    hole, descends at 2 cm/s until the peg would be 25 mm inside, dwells 0.5 s, lifts. The chamfer (6 mm, 45 degrees) catches
    the peg only if the controller yields sideways."""
    assert PLANT.tasks
    (x_lo, x_hi), (y_lo, y_hi) = PLANT.task_xy
    variant = "c075" if rng.uniform() < 0.5 else "c150"
    fixture_pos = np.array([rng.uniform(x_lo + 0.02, x_hi - 0.02), rng.uniform(y_lo + 0.03, y_hi - 0.03), R.TABLE_TOP_Z])
    fx = PEG_FIXTURES[variant]
    hole_top = R.TABLE_TOP_Z + fx.depth
    e = float(rng.uniform(0.002, 0.006)); ang = rng.uniform(0, 2 * np.pi)
    aim = fixture_pos[:2] + e * np.array([np.cos(ang), np.sin(ang)])
    peg_len = 2 * PEG.size[2]
    above = np.array([aim[0], aim[1], hole_top + 0.03 + peg_len])  # the tip 3 cm above the hole
    insert = np.array([aim[0], aim[1], hole_top - 0.025 + peg_len])  # the tip 25 mm inside, if it gets there
    start = above + np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.03])
    pts = [start, above, insert, above]
    speeds = [0.15 * speed, 0.02 * speed, 0.1 * speed]
    dwell = [0.3, 0.5, 0.3]
    path, times = R._tool_waypoints(pts, speeds, dwell, dt)
    labels = ["approach", "approach", "insert", "insert", "lift", "free"]
    tilt, a2 = np.radians(rng.uniform(0.0, 2.0)), rng.uniform(0, 2 * np.pi)
    axis = np.array([np.sin(tilt) * np.cos(a2), np.sin(tilt) * np.sin(a2), -np.cos(tilt)])
    meta = {"task": {"name": "peg", "scene": str(peg_scene(variant)), "object": PEG.name, "weld": True, "weld_start": True, "release_time": 1e9,
                     "fixture": fx.name, "fixture_pos": fixture_pos.tolist(), "hole_top_z": float(hole_top), "hole_center": fixture_pos[:2].tolist(),
                     "peg_half_len": PEG.size[2], "target_depth": 0.025, "clearance": float(fx.hole_half - PEG.size[0]), "lateral_error": e, "variant": variant},
            "speed": speed, "normal": [0.0, 0.0, 1.0], "surface": None}
    ref = _reference(path, times, labels, nominal, axis, rng, "peg", meta, dt)
    if ref is None:
        return None
    ref.meta["normal"] = [0.0, 0.0, 1.0]
    ref.meta["oracle_plane"] = [0.0, 0.0, hole_top + peg_len]  # the tool point's plane when the hanging peg's tip reaches the hole's top plane
    ref.meta["oracle_soft_tangent"] = True  # while blocked above the hole the lateral stiffness is weak: the chamfer guides the peg
    ref.meta["oracle_contact_eps"] = 0.01
    ref.meta["task"]["object_pos"], ref.meta["task"]["object_quat"] = _held_object_pose(nominal, ref.q[0], PEG.size[2])
    k_in = min(len(ref.q) - 1, int(times[3] / dt))
    if not R._links_clear(nominal, [(ref.q[k_in], 0.0)], pressed=None):
        return None
    return ref


def drawer(rng: np.random.Generator, nominal: NominalModel, speed: float, dt: float = 1e-3) -> R.Reference | None:
    """A drawer (protocol v5): a panel on a slide joint with friction, open by 10 cm; the tool pushes its front closed to the
    stop along a reference that drifts 2 cm sideways over the travel and continues 2 cm past the stop."""
    assert PLANT.tasks
    (x_lo, x_hi), (y_lo, y_hi) = PLANT.task_xy
    ref = None
    for _ in range(8):  # directions until a shallow tool axis reaches (a steep one leads with the hand, not the fingertips)
        base_pos = np.array([rng.uniform(x_lo, x_hi - 0.04), rng.uniform(y_lo + 0.02, y_hi - 0.02), R.TABLE_TOP_Z + DRAWER.size[2] + 0.002])
        ang = rng.uniform(-np.radians(100), np.radians(100))  # the slide axis within 100 degrees of +x (pushing away from the base)
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
        side = np.array([-np.sin(ang), np.cos(ang), 0.0])
        base_quat = [float(np.cos(ang / 2)), 0.0, 0.0, float(np.sin(ang / 2))]
        face = base_pos - d * DRAWER.size[0]
        face[2] = R.TABLE_TOP_Z + 0.035  # the contact point on the lower part of the front
        drift = 0.02 * (1.0 if rng.uniform() < 0.5 else -1.0) * side
        pts = [face - 0.08 * d, face, face + 0.12 * d + drift, face + 0.07 * d + drift]
        speeds = [0.1 * speed, 0.05 * speed, 0.1 * speed]
        dwell = [0.3, 0.4, 0.3]
        path, times = R._tool_waypoints(pts, speeds, dwell, dt)
        labels = ["approach", "approach", "push", "push", "retreat", "retreat"]
        meta = {"task": {"name": "drawer", "scene": str(drawer_scene()), "object": DRAWER.name, "object_q0": 0.0, "base_pos": base_pos.tolist(), "base_quat": base_quat,
                         "axis": d.tolist(), "travel": float(DRAWER.range[1]), "weld": False, "push_dir": d.tolist()},
                "speed": speed, "normal": (-d).tolist(), "surface": None, "oracle_plane": (face + DRAWER.range[1] * d).tolist()}  # the oracle's plane: the front's face at the stop
        for tilt in (PUSH_TILT, np.radians(40)):
            axis = np.array([np.cos(tilt) * d[0], np.cos(tilt) * d[1], -np.sin(tilt)])
            meta["task"]["tilt_deg"] = float(np.degrees(tilt))
            ref = _reference(path, times, labels, nominal, axis, rng, "drawer", meta, dt)
            if ref is not None:
                break
        if ref is not None:
            break
    if ref is None:
        return None
    ref.meta["normal"], ref.meta["oracle_plane"] = (-d).tolist(), meta["oracle_plane"]
    ref.meta["oracle_contact_eps"] = 0.02  # the fingertips stop a few millimetres short of the nominal stop plane
    if not R._links_clear(nominal, [(ref.q[min(len(ref.q) - 1, int(times[i] / dt))], 0.0) for i in (1, 3)], pressed=None):
        return None
    return ref


def door(rng: np.random.Generator, nominal: NominalModel, speed: float, dt: float = 1e-3) -> R.Reference | None:
    """A door (protocol v5): a panel on a vertical hinge at its edge, open by 40 degrees toward the robot; the tool pushes it
    closed along a straight line from the panel's face (15 cm from the hinge) to where that point lies when closed, plus 1 cm,
    the contact sliding on the face as the angle changes."""
    assert PLANT.tasks
    (x_lo, x_hi), (y_lo, y_hi) = PLANT.task_xy
    q0, r = DOOR_OPEN, 0.15
    ref = None
    for _ in range(8):
        hinge = np.array([rng.uniform(x_lo + 0.02, x_hi - 0.02), rng.uniform(y_lo, y_lo + 0.05), R.TABLE_TOP_Z + DOOR.size[2] + 0.002])
        p_open = hinge + r * np.array([-np.sin(q0), np.cos(q0), 0.0])  # the contact point on the open panel (the body's x axis at yaw 90 degrees plus q)
        p_closed = hinge + r * np.array([0.0, 1.0, 0.0])
        p_open[2] = p_closed[2] = R.TABLE_TOP_Z + 0.035  # the contact point on the lower part of the panel
        u0 = np.array([np.cos(q0), np.sin(q0), 0.0])  # the push direction at the start: into the panel's face toward the robot
        pts = [p_open - 0.08 * u0, p_open, p_closed + np.array([0.01, 0.0, 0.0]), p_closed - 0.05 * u0]
        speeds = [0.1 * speed, 0.05 * speed, 0.1 * speed]
        dwell = [0.3, 0.4, 0.3]
        path, times = R._tool_waypoints(pts, speeds, dwell, dt)
        labels = ["approach", "approach", "push", "push", "retreat", "retreat"]
        base_quat = [float(np.cos(np.pi / 4)), 0.0, 0.0, float(np.sin(np.pi / 4))]  # yaw 90 degrees: the closed panel lies along +y
        meta = {"task": {"name": "door", "scene": str(door_scene()), "object": DOOR.name, "object_q0": float(q0), "base_pos": hinge.tolist(), "base_quat": base_quat,
                         "hinge": hinge.tolist(), "r": r, "target_angle": 0.0, "weld": False, "push_dir": u0.tolist()},
                "speed": speed, "normal": [-1.0, 0.0, 0.0], "surface": None, "oracle_plane": (p_closed - np.array([0.015, 0.0, 0.0])).tolist()}  # the oracle's plane: 15 mm before the closed panel's face (the fingertips stop short of it, and the engagement precedes the stop)
        for tilt in (PUSH_TILT, np.radians(40)):
            axis = np.array([np.cos(tilt) * u0[0], np.cos(tilt) * u0[1], -np.sin(tilt)])
            meta["task"]["tilt_deg"] = float(np.degrees(tilt))
            ref = _reference(path, times, labels, nominal, axis, rng, "door", meta, dt)
            if ref is not None:
                break
        if ref is not None:
            break
    if ref is None:
        return None
    ref.meta["normal"], ref.meta["oracle_plane"] = [-1.0, 0.0, 0.0], meta["oracle_plane"]
    ref.meta["oracle_contact_eps"] = 0.02
    if not R._links_clear(nominal, [(ref.q[min(len(ref.q) - 1, int(times[i] / dt))], 0.0) for i in (1, 3)], pressed=None):
        return None
    return ref
