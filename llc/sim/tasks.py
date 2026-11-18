"""Task scenes for Milestone 9 (ADR 0009) and Milestone 12 (ADR 0012): the plant scene plus task objects, built
with MjSpec and written next to the plant scenes. Objects are bodies defined after the robot, so the arm keeps
the first seven DoF and ``PandaSim`` slices them: free boxes (pushing, the pick, erasing), jointed panels (a
drawer on a slide joint, a door on a hinge) and static fixtures (the peg's block with a chamfered square hole).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from llc.sim.plant import ASSETS, PLANT


@dataclass(frozen=True)
class BoxObject:
    name: str = "box"
    size: tuple[float, float, float] = (0.025, 0.025, 0.025)  # half-sizes (m): a 5 cm cube
    mass: float = 0.2  # kg
    friction: float = 0.6
    rgba: tuple[float, float, float, float] = (0.85, 0.45, 0.2, 1.0)


@dataclass(frozen=True)
class JointedObject:
    """A panel on one joint (ADR 0012): a drawer front on a slide joint, a door on a hinge. The body's base pose is set per
    case (``PandaSim.set_object_base``); the panel geom is offset from the joint by ``geom_pos`` (a door's hinge at its edge)."""
    name: str
    size: tuple[float, float, float]  # half-sizes of the panel
    joint: str  # "slide" or "hinge"
    axis: tuple[float, float, float]
    range: tuple[float, float]
    mass: float = 1.0
    friction: float = 0.6
    damping: float = 10.0
    frictionloss: float = 2.0
    geom_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgba: tuple[float, float, float, float] = (0.55, 0.6, 0.7, 1.0)


@dataclass(frozen=True)
class Fixture:
    """A static block with a square hole and a 45 degree chamfer (ADR 0012, the peg family): four walls around the hole and
    four tilted plates on top; the body has no joint and its geoms count as environment in ``PandaSim``."""
    name: str = "fixture"
    hole_half: float = 0.01075  # m, half-width of the hole (the 20 mm peg has 0.75 mm of clearance per side)
    outer_half: float = 0.05
    depth: float = 0.03  # m, the block's height (the hole's depth)
    chamfer: float = 0.006  # m, the chamfer's width (and height: 45 degrees)
    friction: float = 0.4
    rgba: tuple[float, float, float, float] = (0.75, 0.72, 0.65, 1.0)


_CONTACT = {"solref": np.array([-20000.0, -200.0]), "solimp": np.array([0.3, 0.95, 0.005, 0.5, 2.0])}


def _add_box(body, name, size, pos, rgba, friction, mass=None, quat=None):
    geom = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=np.array(size, float), pos=np.array(pos, float), rgba=np.array(rgba))
    if mass is not None:
        geom.mass = mass
    if quat is not None:
        geom.quat = np.array(quat, float)
    geom.friction = np.array([friction, 0.005, 0.0001])
    geom.solref, geom.solimp = _CONTACT["solref"], _CONTACT["solimp"]
    return geom


def build_task_scene(objects: tuple, tag: str, base: Path | None = None, weld: str | None = None, weld_offset=(0.0, 0.0, 0.0)) -> Path:
    """The plant scene with the task objects (task code places them) and no keyframe. ``weld`` names an object
    held by the tool: a weld constraint ``hold`` fixes it at the tool point plus ``weld_offset`` (tool frame; the peg
    hangs beyond the fingertips) (rollouts switch it off to release)."""
    base = base or PLANT.scene
    out = ASSETS / f"{base.stem}_task_{tag}.xml"
    if out.exists():
        return out
    spec = mujoco.MjSpec.from_file(str(base))
    for key in list(spec.keys):
        spec.delete(key)
    if weld is not None:
        tcp = np.array(spec.site("tool_tip").pos) + np.asarray(weld_offset, float)
        eq = spec.add_equality(name="hold", type=mujoco.mjtEq.mjEQ_WELD, name1="tool", name2=weld, objtype=mujoco.mjtObj.mjOBJ_BODY)
        eq.data = np.concatenate([[0.0, 0.0, 0.0], tcp, [1.0, 0.0, 0.0, 0.0], [1.0, 0.0]])  # anchor in the object's frame (its centre), the object's pose relative to the tool, torque scale
    n_q = 0
    for obj in objects:
        if isinstance(obj, BoxObject):
            body = spec.worldbody.add_body(name=obj.name, pos=np.array([0.45, 0.0, 0.5]))
            body.add_freejoint()
            _add_box(body, obj.name, obj.size, (0, 0, 0), obj.rgba, obj.friction, mass=obj.mass)
            n_q += 7
        elif isinstance(obj, JointedObject):
            body = spec.worldbody.add_body(name=obj.name, pos=np.array([0.45, 0.0, 0.3]))
            jnt = body.add_joint(name=obj.name, type=mujoco.mjtJoint.mjJNT_SLIDE if obj.joint == "slide" else mujoco.mjtJoint.mjJNT_HINGE, axis=np.array(obj.axis, float))
            jnt.range = np.array(obj.range, float); jnt.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            jnt.damping = np.array([obj.damping, 0.0, 0.0]); jnt.frictionloss = obj.frictionloss  # MuJoCo 4: damping is per DoF (3)
            _add_box(body, obj.name, obj.size, obj.geom_pos, obj.rgba, obj.friction, mass=obj.mass)
            n_q += 1
        elif isinstance(obj, Fixture):
            body = spec.worldbody.add_body(name=obj.name, pos=np.array([0.45, 0.0, 0.3]))
            h, o, d, c = obj.hole_half, obj.outer_half, obj.depth, obj.chamfer
            w = o - h
            for s, i in ((1, 0), (-1, 1)):  # the walls of the hole: along x (full width) and along y (between the x walls)
                _add_box(body, f"{obj.name}_x{i}", (w / 2, o, d / 2), (s * (h + w / 2), 0.0, d / 2), obj.rgba, obj.friction)
                _add_box(body, f"{obj.name}_y{i}", (h, w / 2, d / 2), (0.0, s * (h + w / 2), d / 2), obj.rgba, obj.friction)
            t = 0.0015  # half-thickness of the chamfer plates; their inner faces run from the hole's edge at the top of the block outward and up at 45 degrees
            ang = np.radians(22.5)
            for s in (1, -1):
                cx, cz = h + c / 2 + t / np.sqrt(2), d + c / 2 - t / np.sqrt(2)
                _add_box(body, f"{obj.name}_cx{'p' if s > 0 else 'n'}", (c / np.sqrt(2), o, t), (s * cx, 0.0, cz), obj.rgba, obj.friction, quat=(np.cos(ang), 0.0, -s * np.sin(ang), 0.0))
                _add_box(body, f"{obj.name}_cy{'p' if s > 0 else 'n'}", (o, c / np.sqrt(2), t), (0.0, s * cx, cz), obj.rgba, obj.friction, quat=(np.cos(ang), s * np.sin(ang), 0.0, 0.0))
        else:
            raise TypeError(type(obj))
    model = spec.compile()
    from llc.sim.panda import NUM_JOINTS

    assert model.nu == NUM_JOINTS and model.nq == NUM_JOINTS + n_q, (model.nu, model.nq, n_q)
    import os

    rel = os.path.relpath(ASSETS, out.parent)
    out.write_text(spec.to_xml().replace(str(ASSETS) + "/", "" if rel == "." else rel + "/"))
    return out
