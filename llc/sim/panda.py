"""MuJoCo simulation of the torque-controlled Panda with a fixed tool.

The simulator plays the role of the physical robot. Conventional controllers
never read this model's parameters; they use ``NominalModel`` instead, so that
dynamics perturbations (payload, friction, delay, noise) stay hidden from the
controller as they would be on hardware.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from llc.sim.plant import PLANT

SCENE_PATH = PLANT.scene  # ADR 0009: LLC_PLANT selects the sphere-tool plant (v2), the gripper plant (v3) or another robot (ADR 0011)
NUM_JOINTS = PLANT.n_joints  # the selected robot's constants under their historical names (ADR 0011): 7 joints, 87/12 Nm, Franka datasheet velocities on the Panda
TORQUE_LIMITS = np.array(PLANT.torque_limits)
VELOCITY_LIMITS = np.array(PLANT.velocity_limits)
HOME = np.array(PLANT.home)
ENV_GEOMS = PLANT.env_geoms
TOOL_GEOMS = PLANT.tool_geoms


# Compliant surfaces (ADR 0008): with a constant impedance (solimp d = 0.5 over a 10 cm width) the contact
# behaves as a linear spring at the tool with about 21 N/m per unit of direct solref stiffness (calibrated
# on blocked and contact cases: per-step force/sink ratios within +-20% of the mean, sinks up to 2 cm).
SURFACE_STIFFNESS_PER_SOLREF = 21.0
SURFACE_SOLIMP = (0.5, 0.5, 0.1)


def surface_params(stiffness: float, zeta: float = 0.7) -> dict:
    """``DynamicsPerturbation`` fields for a table and wall of effective stiffness ``stiffness`` (N/m)
    at the tool, damping ratio ``zeta`` (the scene's own surface has about 0.7)."""
    k = stiffness / SURFACE_STIFFNESS_PER_SOLREF
    return {"surface_solref": (-float(k), -float(2.0 * zeta * np.sqrt(k))), "surface_solimp": SURFACE_SOLIMP}


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_PATH))


@dataclass
class DynamicsPerturbation:
    """Deviations of the simulated robot from the nominal model (all zero = nominal)."""

    payload_mass: float = 0.0  # kg, point mass rigidly attached to the tool
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.10)  # m, tool frame
    inertia_scale: float = 1.0  # multiplies every link's rotational inertia
    friction_loss: float = 0.0  # Nm, dry friction torque on every joint
    damping_scale: float = 1.0  # multiplies joint viscous damping
    actuation_delay: float = 0.0  # s, torque command transport delay
    obs_noise_q: float = 0.0  # rad, std of position measurement noise
    obs_noise_qd: float = 0.0  # rad/s, std of velocity measurement noise
    # Robot-instance deviations (one entry per moving body link1..link7 and the tool, 8 entries),
    # modelling an imperfectly identified robot; all ones / zeros = nominal.
    link_mass_scale: tuple[float, ...] = (1.0,) * PLANT.moving_bodies
    link_inertia_scale: tuple[float, ...] = (1.0,) * PLANT.moving_bodies
    link_com_offset: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),) * PLANT.moving_bodies  # m, body frame
    joint_damping_scale: tuple[float, ...] = (1.0,) * PLANT.n_joints
    joint_armature_scale: tuple[float, ...] = (1.0,) * PLANT.n_joints
    torque_scale: tuple[float, ...] = (1.0,) * PLANT.n_joints  # actuator gain error: applied torque = scale * command
    # MuJoCo direct solref (-stiffness, -damping) of the table and wall geoms; (0, 0) keeps the scene's
    # (about 20 kN/m direct, 60 kN/m effective). Compliant-surface groups of ADR 0008.
    surface_solref: tuple[float, float] = (0.0, 0.0)
    surface_solimp: tuple[float, float, float] = (0.0, 0.0, 0.0)  # with surface_solref: (d0, d_width, width) of the surface geoms; zeros keep the scene's
    # Milestone 12 (ADR 0012): the consolidation distribution's extra unknowns
    actuation_lag: float = 0.0  # s, first-order lag of the applied torque after the transport delay (0 = none)
    surface_friction: float = 0.0  # sliding friction of the floor, table and wall geoms (0 = the scene's)
    object_mass_scale: float = 1.0  # multiplies the mass and inertia of every task object (bodies jointed after the arm)

    def is_nominal(self) -> bool:
        return self == DynamicsPerturbation()


@dataclass
class ContactState:
    f_normal: float  # N, total normal force between tool and environment
    f_tangent: float  # N, total tangential force magnitude
    n_tool_contacts: int
    n_other_contacts: int  # contacts of a robot geom other than tool-environment or tool-object (links, knuckles)
    f_object: float = 0.0  # N, total normal force between the tool and task objects (Milestone 9 task scenes)
    n_object_contacts: int = 0
    f_object_env: float = 0.0  # N, total normal force between task objects and the environment (a placed block on the table)


def _in_subtree(m: mujoco.MjModel, b: int, root: int) -> bool:
    while b > 0:
        if b == root:
            return True
        b = int(m.body_parentid[b])
    return b == root


def _quat_from_matrix(r: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, r.reshape(9))
    return q


class PandaSim:
    """The arm's 7 DoF are the first joints of the model (``qpos[:7]``, ``qvel[:7]``, ``ctrl``); a task
    scene (Milestone 9) may add free objects after them, which this class leaves to the task code
    (``set_object_pose``, ``object_pose``)."""

    def __init__(self, perturbation: DynamicsPerturbation | None = None, seed: int = 0, scene: str | Path | None = None):
        self.model = load_model() if scene is None else mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.perturbation = perturbation or DynamicsPerturbation()
        self.rng = np.random.default_rng(seed)
        m = self.model
        self.tool_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tool")
        self.tool_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
        self.touch_sensor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "tool_touch")
        self.env_geom_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ENV_GEOMS}
        self.tool_geom_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in TOOL_GEOMS}
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PLANT.base_link)
        robot_bodies = {b for b in range(m.nbody) if _in_subtree(m, b, base)}  # the base link and everything below it (link0..link7 and the tool on the Panda)
        self.robot_geom_ids = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in robot_bodies}
        static = {b for b in range(1, m.nbody) if b not in robot_bodies and m.body_jntnum[b] == 0 and m.body_dofnum[b] == 0}  # fixtures (ADR 0012): environment
        self.env_geom_ids |= {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in static}
        self.object_geom_ids = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) not in robot_bodies and int(m.geom_bodyid[g]) != 0 and int(m.geom_bodyid[g]) not in static}
        self._apply_perturbation(self.perturbation)
        self._delay_steps = int(round(self.perturbation.actuation_delay / self.dt))
        self._delay_buffer: deque[np.ndarray] = deque()
        lag = float(self.perturbation.actuation_lag)
        self._lag_alpha = self.dt / (lag + self.dt) if lag > 0.0 else 1.0  # ADR 0012: first-order actuator lag after the delay
        self._tau_lag = np.zeros(NUM_JOINTS)
        self._f6 = np.zeros(6)
        self.reset(HOME)

    # ------------------------------------------------------------------ setup
    def _apply_perturbation(self, p: DynamicsPerturbation) -> None:
        m = self.model
        moving = [b for b in range(1, m.nbody) if m.body_jntnum[b] == 1 and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_HINGE][:NUM_JOINTS] + [self.tool_body]  # the jointed links and the tool
        for i, b in enumerate(moving):
            m.body_mass[b] *= p.link_mass_scale[i]
            m.body_inertia[b] *= p.link_inertia_scale[i] * p.link_mass_scale[i]
            m.body_ipos[b] += np.asarray(p.link_com_offset[i], dtype=float)
        if p.inertia_scale != 1.0:
            m.body_inertia[1:] *= p.inertia_scale
        if p.payload_mass > 0.0:
            self._add_point_mass(self.tool_body, p.payload_mass, np.asarray(p.payload_offset, dtype=float))
        m.dof_frictionloss[:NUM_JOINTS] = p.friction_loss  # the arm's DoF; task objects (free joints after them) are untouched
        m.dof_damping[:NUM_JOINTS] *= p.damping_scale * np.asarray(p.joint_damping_scale)
        m.dof_armature[:NUM_JOINTS] *= np.asarray(p.joint_armature_scale)
        m.actuator_gainprm[:, 0] *= np.asarray(p.torque_scale)
        if tuple(p.surface_solref) != (0.0, 0.0):
            for name in ("table", "wall"):  # the surface's own parameters decide the contact (priority over the tool's)
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                m.geom_solref[gid] = p.surface_solref
                if tuple(p.surface_solimp) != (0.0, 0.0, 0.0):
                    m.geom_solimp[gid, :3] = p.surface_solimp
                m.geom_priority[gid] = 2  # above the gripper pads' priority 1 (plant v3) and the sphere's 0
        if p.surface_friction > 0.0:  # ADR 0012: the environment's sliding friction
            for name in ("floor", "table", "wall"):
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    m.geom_friction[gid, 0] = p.surface_friction
        if p.object_mass_scale != 1.0:  # ADR 0012: task objects (bodies jointed after the arm's degrees of freedom)
            for b in range(1, m.nbody):
                if m.body_jntnum[b] > 0 and m.body_dofadr[b] >= NUM_JOINTS:
                    m.body_mass[b] *= p.object_mass_scale
                    m.body_inertia[b] *= p.object_mass_scale
        mujoco.mj_setConst(m, self.data)

    def _add_point_mass(self, body: int, mass: float, offset: np.ndarray) -> None:
        """Merge a point mass at ``offset`` (body frame) into the body's inertial parameters."""
        m = self.model
        m0 = float(m.body_mass[body])
        c0 = m.body_ipos[body].copy()
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, m.body_iquat[body])
        r = r.reshape(3, 3)
        inertia0 = r @ np.diag(m.body_inertia[body]) @ r.T  # about c0, body frame
        m1 = m0 + mass
        c1 = (m0 * c0 + mass * offset) / m1

        def shift(inertia, mass_, d):  # parallel-axis shift of an inertia about COM by displacement d
            return inertia + mass_ * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

        inertia1 = shift(inertia0, m0, c0 - c1) + shift(np.zeros((3, 3)), mass, offset - c1)
        w, v = np.linalg.eigh(inertia1)
        if np.linalg.det(v) < 0:
            v[:, 0] *= -1
        m.body_mass[body] = m1
        m.body_ipos[body] = c1
        m.body_inertia[body] = w
        m.body_iquat[body] = _quat_from_matrix(v)

    # ------------------------------------------------------------------ state
    def reset(self, q: np.ndarray, qd: np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:NUM_JOINTS] = q
        self.data.qvel[:NUM_JOINTS] = 0.0 if qd is None else qd
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._delay_buffer.clear()
        for _ in range(self._delay_steps):
            self._delay_buffer.append(np.zeros(NUM_JOINTS))
        self._tau_lag = np.zeros(NUM_JOINTS)

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[:NUM_JOINTS].copy()

    @property
    def qd(self) -> np.ndarray:
        return self.data.qvel[:NUM_JOINTS].copy()

    # ------------------------------------------------------------------ task objects (free bodies after the arm)
    def set_object_pose(self, body: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Place a free body (its joint's qpos: position then quaternion) at rest."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        jid = int(self.model.body_jntadr[bid])
        adr, vadr = int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])
        self.data.qpos[adr:adr + 3] = pos
        self.data.qpos[adr + 3:adr + 7] = quat
        self.data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def set_object_base(self, body: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Move a jointed object's or a fixture's base frame (ADR 0012: the case places the drawer, the door, the peg's fixture)."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        self.model.body_pos[bid] = pos
        self.model.body_quat[bid] = quat
        mujoco.mj_forward(self.model, self.data)

    def set_object_joint(self, body: str, q0: float) -> None:
        """Set a one-joint object's joint position at rest (the drawer's opening, the door's angle)."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        jid = int(self.model.body_jntadr[bid])
        self.data.qpos[int(self.model.jnt_qposadr[jid])] = q0
        self.data.qvel[int(self.model.jnt_dofadr[jid])] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def try_grasp(self, eq: int, body: str, tol_lateral: float, tol_height: float) -> bool:
        """Activate the weld ``eq`` between the tool and ``body`` at their current relative pose if the body's centre is
        within ``tol_lateral`` of the tool point in the tool's plane and within ``tol_height`` along the tool axis
        (protocol v4: the grasp succeeds only if the block is inside the fingertip gap). Returns the outcome."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        r_tool = self.data.xmat[self.tool_body].reshape(3, 3)
        d = r_tool.T @ (self.data.xpos[bid] - self.data.site_xpos[self.tool_site])
        if np.hypot(d[0], d[1]) > tol_lateral or abs(d[2]) > tol_height:
            return False
        rel_pos = r_tool.T @ (self.data.xpos[bid] - self.data.xpos[self.tool_body])
        q_inv, rel_quat = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q_inv, self.data.xquat[self.tool_body])
        mujoco.mju_mulQuat(rel_quat, q_inv, self.data.xquat[bid])
        self.model.eq_data[eq, 3:6] = rel_pos  # the object's pose in the tool body's frame: nothing snaps at activation
        self.model.eq_data[eq, 6:10] = rel_quat
        self.data.eq_active[eq] = 1
        return True

    def object_pose(self, body: str) -> tuple[np.ndarray, np.ndarray]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()

    @property
    def time(self) -> float:
        return float(self.data.time)

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        """Measured joint position and velocity (with sensor noise if configured)."""
        q, qd = self.data.qpos[:NUM_JOINTS].copy(), self.data.qvel[:NUM_JOINTS].copy()
        p = self.perturbation
        if p.obs_noise_q > 0.0:
            q += self.rng.normal(0.0, p.obs_noise_q, NUM_JOINTS)
        if p.obs_noise_qd > 0.0:
            qd += self.rng.normal(0.0, p.obs_noise_qd, NUM_JOINTS)
        return q, qd

    @property
    def tool_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tool_site].copy()

    @property
    def tool_vel(self) -> np.ndarray:
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, self.tool_site)
        return jac[:, :NUM_JOINTS] @ self.data.qvel[:NUM_JOINTS]

    # ------------------------------------------------------------------ stepping
    def apply_tool_force(self, f_world: np.ndarray) -> None:
        """External force (N, world frame) acting at the tool body's centre of mass during the next step."""
        self.data.xfrc_applied[self.tool_body, :3] = f_world
        self.data.xfrc_applied[self.tool_body, 3:] = 0.0

    def step(self, tau: np.ndarray) -> np.ndarray:
        """Apply a torque command for one physics step. Returns the torque actually applied."""
        tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
        if self._delay_steps:
            self._delay_buffer.append(tau)
            tau = self._delay_buffer.popleft()
        if self._lag_alpha < 1.0:
            self._tau_lag = self._tau_lag + self._lag_alpha * (tau - self._tau_lag)
            tau = self._tau_lag.copy()
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)
        return tau

    # ------------------------------------------------------------------ contact
    def contact(self) -> ContactState:
        d, m = self.data, self.model
        f_n = f_t = f_obj = f_obj_env = 0.0
        n_tool = n_other = n_obj = 0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            tool_env = (g1 in self.tool_geom_ids and g2 in self.env_geom_ids) or (g2 in self.tool_geom_ids and g1 in self.env_geom_ids)
            tool_obj = (g1 in self.tool_geom_ids and g2 in self.object_geom_ids) or (g2 in self.tool_geom_ids and g1 in self.object_geom_ids)
            if tool_env:
                mujoco.mj_contactForce(m, d, i, self._f6)
                f_n += abs(self._f6[0])
                f_t += float(np.hypot(self._f6[1], self._f6[2]))
                n_tool += 1
            elif tool_obj:
                mujoco.mj_contactForce(m, d, i, self._f6)
                f_obj += abs(self._f6[0])
                n_obj += 1
            elif g1 in self.robot_geom_ids or g2 in self.robot_geom_ids:  # object-environment contacts are not the robot's
                n_other += 1
            elif (g1 in self.object_geom_ids and g2 in self.env_geom_ids) or (g2 in self.object_geom_ids and g1 in self.env_geom_ids):
                mujoco.mj_contactForce(m, d, i, self._f6)
                f_obj_env += abs(self._f6[0])
        return ContactState(f_n, f_t, n_tool, n_other, f_obj, n_obj, f_obj_env)

    @property
    def touch_force(self) -> float:
        """Normal contact force on the tool tip from the touch sensor (N)."""
        return float(self.data.sensordata[self.model.sensor_adr[self.touch_sensor]])

    def in_collision(self, q: np.ndarray) -> bool:
        """True if configuration ``q`` has any contact (used for reference generation)."""
        qpos, qvel = self.data.qpos.copy(), self.data.qvel.copy()
        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        hit = self.data.ncon > 0
        self.data.qpos[:], self.data.qvel[:] = qpos, qvel
        mujoco.mj_forward(self.model, self.data)
        return hit
