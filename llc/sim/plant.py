"""Plant selection (ADR 0009). ``v2``: the Panda with the fixed spherical tool of ADR 0001 (the record
of reports 01 to 13); ``v3``: the Panda with the Robotiq 2F-85 baked at a fixed opening, the DROID
robot (Milestone 9). Chosen by the environment variable ``LLC_PLANT``; the scene, the tool geoms,
the tool point's offset from its contact face and the surface-family geometry follow from it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"


@dataclass(frozen=True)
class PlantSpec:
    name: str
    scene: Path
    tool_geoms: tuple[str, ...]  # geoms whose contacts with the environment are the tool's
    tool_offset: float  # m, from the tool point to its contact face along the tool axis (sphere radius; 0 for the fingertip faces)
    table_tilt_deg: tuple[float, float]  # tool-axis tilt from vertical in the table contact family
    wall_tilt_deg: tuple[float, float]  # tool-axis angle below horizontal at the wall
    wall_depth: tuple[float, float]  # m, blocked targets inside the wall
    cache_tag: str  # suffix of cached reference pools


def _v3() -> PlantSpec:
    from llc.sim.gripper import DRIVER_CLOSED, PAD_GEOMS, build_scene

    scene = ASSETS / f"panda_2f85_{int(round(1000 * DRIVER_CLOSED)):04d}.xml"
    if not scene.exists():
        build_scene(DRIVER_CLOSED, scene)
    return PlantSpec("v3", scene, PAD_GEOMS, 0.0, (0.0, 10.0), (0.0, 15.0), (0.03, 0.06), "_v3")


V2 = PlantSpec("v2", ASSETS / "panda_scene.xml", ("tool_flange", "tool_tip"), 0.02, (0.0, 30.0), (30.0, 45.0), (0.03, 0.06), "")


def select(name: str) -> PlantSpec:
    if name == "v2":
        return V2
    if name == "v3":
        return _v3()
    raise ValueError(f"unknown plant {name!r} (LLC_PLANT=v2|v3)")


PLANT = select(os.environ.get("LLC_PLANT", "v2"))
