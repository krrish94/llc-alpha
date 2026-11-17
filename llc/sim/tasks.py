"""Task scenes for Milestone 9 (ADR 0009): the plant scene plus free objects, built with MjSpec and
written next to the plant scenes. Objects are bodies with a free joint defined after the robot, so
the arm keeps the first seven DoF and ``PandaSim`` slices them."""
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


def build_task_scene(objects: tuple[BoxObject, ...], tag: str, base: Path | None = None, weld: str | None = None) -> Path:
    """The plant scene with free boxes (task code places them) and no keyframe. ``weld`` names an object
    held by the tool: a weld constraint ``hold`` fixes it at the tool point (rollouts switch it off to release)."""
    base = base or PLANT.scene
    out = ASSETS / f"{base.stem}_task_{tag}.xml"
    if out.exists():
        return out
    spec = mujoco.MjSpec.from_file(str(base))
    for key in list(spec.keys):
        spec.delete(key)
    if weld is not None:
        tcp = np.array(spec.site("tool_tip").pos)
        eq = spec.add_equality(name="hold", type=mujoco.mjtEq.mjEQ_WELD, name1="tool", name2=weld, objtype=mujoco.mjtObj.mjOBJ_BODY)
        eq.data = np.concatenate([[0.0, 0.0, 0.0], tcp, [1.0, 0.0, 0.0, 0.0], [1.0, 0.0]])  # anchor in the object's frame (its centre), the object's pose relative to the tool (at the tool point, aligned), torque scale
    for obj in objects:
        body = spec.worldbody.add_body(name=obj.name, pos=np.array([0.45, 0.0, 0.5]))
        body.add_freejoint()
        geom = body.add_geom(name=obj.name, type=mujoco.mjtGeom.mjGEOM_BOX, size=np.array(obj.size), mass=obj.mass, rgba=np.array(obj.rgba))
        geom.friction = np.array([obj.friction, 0.005, 0.0001])
        geom.solref = np.array([-20000.0, -200.0])
        geom.solimp = np.array([0.3, 0.95, 0.005, 0.5, 2.0])
    model = spec.compile()
    assert model.nu == 7 and model.nq == 7 + 7 * len(objects), (model.nu, model.nq)
    import os

    rel = os.path.relpath(ASSETS, out.parent)
    out.write_text(spec.to_xml().replace(str(ASSETS) + "/", "" if rel == "." else rel + "/"))
    return out
