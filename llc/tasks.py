"""Task families of Milestone 9 (ADR 0009): box pushing and pick-and-place on the gripper plant.

A scripted task policy emits a tool path from the randomised object pose; the reference is its
joint-space tracking target (IK as for the surface families); the object lives in a task scene
(``llc/sim/tasks.py``) that the rollout loads and resets from the case's task description. The
gripper is not controlled: pushing uses the closed gripper, pick-and-place a gripper baked at a 38 mm
opening holding a 34 mm block by a weld constraint that the rollout releases at the place point.
"""
from __future__ import annotations

import numpy as np

from llc import references as R
from llc.sim.gripper import build_scene
from llc.sim.nominal import NominalModel
from llc.sim.panda import HOME, VELOCITY_LIMITS
from llc.sim.plant import ASSETS, PLANT
from llc.sim.tasks import BoxObject, build_task_scene

PUSH_BOX = BoxObject("box", (0.025, 0.025, 0.025), 0.2, 0.6, (0.85, 0.45, 0.2, 1.0))
PICK_BLOCK = BoxObject("block", (0.017, 0.017, 0.02), 0.3, 0.6, (0.2, 0.4, 0.85, 1.0))
PICK_DRIVER = 0.45  # rad: a 38 mm fingertip gap around the 34 mm block
PUSH_DIST = 0.10  # m
PUSH_TILT = np.radians(25.0)  # tool axis below horizontal while pushing: keeps the wrist above the table
TASK_REGIMES = ("push", "pick")


def push_scene():
    return build_task_scene((PUSH_BOX,), "push")


def pick_scene():
    base = ASSETS / f"panda_2f85_{int(round(1000 * PICK_DRIVER)):04d}.xml"
    if not base.exists():
        build_scene(PICK_DRIVER, base)
    return build_task_scene((PICK_BLOCK,), "pick", base=base, weld=PICK_BLOCK.name)


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
    assert PLANT.name == "v3", "the task families live on the gripper plant"
    for _ in range(10):
        ang = rng.uniform(-np.radians(110), np.radians(110))  # push directions within 110 degrees of +x (pushing toward the base is not reachable with the tool along the push)
        d = np.array([np.cos(ang), np.sin(ang), 0.0])
        box = np.array([rng.uniform(0.40, 0.50), rng.uniform(-0.12, 0.12), R.TABLE_TOP_Z + PUSH_BOX.size[2]])
        end = box + d * PUSH_DIST
        if 0.38 <= end[0] <= 0.57 and abs(end[1]) <= 0.24 and 0.38 <= (box - d * 0.105)[0] <= 0.57 and abs((box - d * 0.105)[1]) <= 0.24:
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
    """Carry the held block to a target on the table, descend, place (the weld is released 0.2 s into
    the dwell), lift. The block's centre is at the tool point; placing puts its bottom 2 mm into the table."""
    assert PLANT.name == "v3", "the task families live on the gripper plant"
    target = np.array([rng.uniform(0.40, 0.52), rng.uniform(-0.15, 0.15), R.TABLE_TOP_Z + PICK_BLOCK.size[2]])
    start = target + np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 0.15])
    start[0] = np.clip(start[0], 0.35, 0.58)
    above = target + np.array([0.0, 0.0, 0.12])
    place = target - np.array([0.0, 0.0, 0.002])
    pts = [start, above, place, above]
    speeds = [0.3 * speed, 0.08 * speed, 0.2 * speed]
    dwell = [0.2, 0.6, 0.3]
    path, times = R._tool_waypoints(pts, speeds, dwell, dt)
    labels = ["transport", "transport", "descend", "place", "lift", "lift"]
    axis = np.array([0.0, 0.0, -1.0])
    q0, err = R._ik_multistart(nominal, start, HOME, axis, rng)
    if err > 1e-3:
        return None
    tcp, rot = nominal.tool_frame(q0)
    quat = np.zeros(4)
    import mujoco

    mujoco.mju_mat2Quat(quat, rot.reshape(9))
    meta = {"task": {"name": "pick", "scene": str(pick_scene()), "object": PICK_BLOCK.name, "object_pos": tcp.tolist(), "object_quat": quat.tolist(),
                     "target": target.tolist(), "weld": True, "release_time": float(times[3] + 0.2)}, "speed": speed}
    ref = _reference(path, times, labels, nominal, axis, rng, "pick", meta, dt)
    if ref is None:
        return None
    if not R._links_clear(nominal, [(ref.q[min(len(ref.q) - 1, int(times[3] / dt))], 0.002)], pressed="table"):
        return None
    return ref
