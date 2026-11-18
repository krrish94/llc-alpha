"""Plant selection (ADR 0009, ADR 0011). ``v2``: the Panda with the fixed spherical tool of ADR 0001 (the
record of reports 01 to 13); ``v3``: the Panda with the Robotiq 2F-85 baked at a fixed opening, the DROID
robot (Milestone 9); ``yam``: the i2rt YAM with its gripper baked closed (Milestone 11). Chosen by the
environment variable ``LLC_PLANT``. The specification carries everything that is the robot's and not the
method's: the scene, the controlled joints and their limits, the home configuration, the tool, the
environment geometry the reference families sample in, the force scale of the protocol (ADR 0011,
decision 7d) and the tuning bounds of the classical gains. ``llc.sim.panda`` exports the selected robot's
constants under their historical names.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    # --- the robot (ADR 0011; the Panda's values are those of reports 01 to 18)
    robot: str = "panda"
    n_joints: int = 7
    torque_limits: tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    velocity_limits: tuple[float, ...] = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
    home: tuple[float, ...] = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)
    base_link: str = "link0"  # the robot's first (fixed) body; every body below it is the robot's
    env_geoms: tuple[str, ...] = ("floor", "table", "wall")
    # --- the environment the reference families sample in (m)
    table_top_z: float = 0.15
    wall_face_x: float = 0.66
    workspace_lo: tuple[float, float, float] = (0.30, -0.35, 0.22)
    workspace_hi: tuple[float, float, float] = (0.56, 0.35, 0.65)
    table_xy: tuple[tuple[float, float], tuple[float, float]] = ((0.38, 0.54), (-0.2, 0.2))  # base points of table presses
    wall_yz: tuple[tuple[float, float], tuple[float, float]] = ((-0.15, 0.15), (0.25, 0.45))  # base points of wall presses
    task_xy: tuple[tuple[float, float], tuple[float, float]] = ((0.40, 0.52), (-0.15, 0.15))  # object positions of the task families
    q1_train: tuple[float, float] = (-2.6, 1.2)  # base-yaw range of the free-space families; the held-out workspace is the rest
    q1_heldout: tuple[float, float] = (1.2, 2.6)
    joint_margin: float = 0.15  # rad from the joint limits when sampling configurations
    tool_min_z: float = 0.08  # m, the tool stays above this height in free-space references
    blocked_tilt_deg: tuple[float, float] = (0.0, 0.0)  # tool-axis tilt from vertical in the table branch of the blocked and task families (the Panda points straight down)
    approach_height: tuple[float, float] = (0.13, 0.25)  # m above the table top: the free-space points of the blocked family's table branch
    task_start_height: tuple[float, float] = (0.15, 0.25)  # m above the table top: the start of the task family
    # --- the force scale of the protocol (ADR 0011, decision 7d)
    force_cap: float = 15.0  # N, sustained-force cap
    reflex_force: float = 25.0  # N, filtered force at which a collision reaction would trigger
    press_floor: float = 7.0  # N, below which the press deficit counts (protocol v3 and later)
    press_depth: tuple[float, float] = (0.004, 0.014)  # m, commanded press depth (7 to 12 N under 5 N + 500 N/m x depth)
    press_min: float = 5.0  # N, the press law's offset (the oracle's floor and the reward's)
    pulse_force: float = 10.0  # N, the hold family's disturbance pulse
    payloads_eval: tuple[float, ...] = (0.0, 1.0, 2.0)  # kg, evaluation payloads
    payload_train_max: float = 3.0  # kg
    payload_heavy: float = 3.0  # kg, the heavy-payload group
    # --- classical gain tuning (CMA-ES): initial stiffness per joint and bounds
    tune_kp_init: tuple[float, ...] = (400.0, 400.0, 300.0, 300.0, 100.0, 60.0, 30.0)
    tune_kp_bounds: tuple[tuple[float, float], ...] = ((5.0, 3000.0),) * 4 + ((2.0, 600.0),) * 3
    tasks: bool = False  # the task families exist on this plant (a gripper with an opening)
    task_scene_builder: str = ""  # "module:function" building the robot's scene at a fingertip opening (m) for the task families

    @property
    def moving_bodies(self) -> int:  # jointed links plus the tool body
        return self.n_joints + 1


def _v3() -> PlantSpec:
    from llc.sim.gripper import DRIVER_CLOSED, PAD_GEOMS, build_scene

    scene = ASSETS / f"panda_2f85_{int(round(1000 * DRIVER_CLOSED)):04d}.xml"
    if not scene.exists():
        build_scene(DRIVER_CLOSED, scene)
    return PlantSpec("v3", scene, PAD_GEOMS, 0.0, (0.0, 10.0), (0.0, 15.0), (0.03, 0.06), "_v3", tasks=True, task_scene_builder="llc.sim.gripper:task_scene")


V2 = PlantSpec("v2", ASSETS / "panda_scene.xml", ("tool_flange", "tool_tip"), 0.02, (0.0, 30.0), (30.0, 45.0), (0.03, 0.06), "")


def _yam() -> PlantSpec:
    """The i2rt YAM (Menagerie model, MIT licence) with its gripper baked closed (ADR 0011, decision 8). Torque
    limits are the motors' peak torques (DM4340 28 Nm, DM4310 10 Nm); the force scale follows the rule of
    protocol v4 (cap 12 N, press band 5 to 8 N, payloads to 1 kg); the environment is the Panda's scaled to
    the arm's 0.78 m reach (table top at 0.06 m, wall face at 0.58 m); the wrist points downward only close to the base, so the
    table region is nearer and the table family's tilt range wider (0 to 25 degrees)."""
    from llc.sim.yam import TOOL_GEOMS as yam_tool_geoms, build_scene

    scene = ASSETS / "yam_0000.xml"
    if not scene.exists():
        build_scene(0.0, scene)
    return PlantSpec("yam", scene, yam_tool_geoms, 0.0, (0.0, 25.0), (0.0, 15.0), (0.02, 0.05), "_yam", robot="yam", n_joints=6,
                     torque_limits=(28.0, 28.0, 28.0, 10.0, 10.0, 10.0), velocity_limits=(10.0, 10.0, 10.0, 20.0, 20.0, 20.0),
                     home=(0.05, 1.73, 1.79, -1.34, 0.0, 0.82), base_link="arm",
                     table_top_z=0.06, wall_face_x=0.58, workspace_lo=(0.22, -0.30, 0.12), workspace_hi=(0.48, 0.30, 0.50),
                     table_xy=((0.27, 0.40), (-0.12, 0.12)), wall_yz=((-0.12, 0.12), (0.18, 0.38)), task_xy=((0.28, 0.38), (-0.10, 0.10)),
                     q1_train=(-2.3, 1.2), q1_heldout=(1.2, 2.6), joint_margin=0.15, tool_min_z=0.05,
                     blocked_tilt_deg=(10.0, 25.0), approach_height=(0.06, 0.14), task_start_height=(0.06, 0.14),
                     force_cap=12.0, reflex_force=20.0, press_floor=5.0, press_depth=(0.0, 0.006), press_min=5.0, pulse_force=5.0,
                     payloads_eval=(0.0, 0.5, 1.0), payload_train_max=1.0, payload_heavy=1.5,
                     tune_kp_init=(80.0, 80.0, 80.0, 10.0, 10.0, 10.0), tune_kp_bounds=((5.0, 600.0),) * 3 + ((1.0, 150.0),) * 3,
                     tasks=True, task_scene_builder="llc.sim.yam:task_scene")


def select(name: str) -> PlantSpec:
    if name == "v2":
        return V2
    if name == "v3":
        return _v3()
    if name == "yam":
        return _yam()
    raise ValueError(f"unknown plant {name!r} (LLC_PLANT=v2|v3|yam)")


PLANT = select(os.environ.get("LLC_PLANT", "v2"))
