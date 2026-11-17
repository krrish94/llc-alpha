"""Nominal (unperturbed) robot model used by conventional controllers and reference generation.

This is the model a control engineer would have: the datasheet robot without payload,
friction or delay. Conventional controllers compute gravity compensation, inverse
dynamics and inverse kinematics from it, never from the simulated plant.
"""
from __future__ import annotations

import mujoco
import numpy as np

from llc.sim.panda import HOME, NUM_JOINTS, load_model


class NominalModel:
    def __init__(self):
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.tool_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
        self.tool_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tool")
        self.joint_range = self.model.jnt_range.copy()
        self._jac = np.zeros((3, NUM_JOINTS))
        self._jacr = np.zeros((3, NUM_JOINTS))
        self._full_m = np.zeros((NUM_JOINTS, NUM_JOINTS))

    def _set(self, q: np.ndarray, qd: np.ndarray | None = None) -> None:
        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0 if qd is None else qd
        mujoco.mj_forward(self.model, self.data)

    def gravity(self, q: np.ndarray) -> np.ndarray:
        """Generalized gravity torque g(q) (Nm)."""
        self._set(q)
        return self.data.qfrc_bias.copy()

    def bias(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Coriolis, centrifugal and gravity torque C(q,qd) qd + g(q) (Nm)."""
        self._set(q, qd)
        return self.data.qfrc_bias.copy()

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        self._set(q)
        mujoco.mj_fullM(self.model, self.data, self._full_m)
        return self._full_m.copy()

    def tool_pos(self, q: np.ndarray) -> np.ndarray:
        self._set(q)
        return self.data.site_xpos[self.tool_site].copy()

    def tool_frame(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool-tip position and the tool body's rotation matrix (world frame)."""
        self._set(q)
        return self.data.site_xpos[self.tool_site].copy(), self.data.xmat[self.tool_body].reshape(3, 3).copy()

    def tool_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positional and rotational Jacobians (3x7 each) of the tool tip."""
        self._set(q)
        mujoco.mj_jacSite(self.model, self.data, self._jac, self._jacr, self.tool_site)
        return self._jac.copy(), self._jacr.copy()

    def in_collision(self, q: np.ndarray) -> bool:
        self._set(q)
        return self.data.ncon > 0

    def link_contacts(self, q: np.ndarray, env_geom: str = "wall", margin: float = 0.02) -> list[str]:
        """Names of robot geoms other than the tool that are within ``margin`` of ``env_geom`` at ``q``
        (the surface families must block the tool, not the hand or the forearm; ADR 0007)."""
        import mujoco

        from llc.sim.panda import TOOL_GEOMS

        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, env_geom)
        old = float(self.model.geom_margin[gid])
        self.model.geom_margin[gid] = margin
        try:
            self._set(q)
            names = []
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if gid not in (c.geom1, c.geom2):
                    continue
                other = c.geom2 if c.geom1 == gid else c.geom1
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other) or str(other)
                if name not in TOOL_GEOMS:
                    names.append(name)
            return names
        finally:
            self.model.geom_margin[gid] = old
            self._set(q)

    def link_clearance(self, q: np.ndarray, env_geom: str = "table", distmax: float = 0.05) -> float:
        """Smallest signed distance (m, negative when penetrating, capped at ``distmax``) between
        ``env_geom`` and any colliding robot geom other than the tool at ``q`` (ADR 0009)."""
        import mujoco

        from llc.sim.panda import TOOL_GEOMS

        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, env_geom)
        self._set(q)
        best = distmax
        fromto = np.zeros(6)
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == 0 or (self.model.geom_contype[g] == 0 and self.model.geom_conaffinity[g] == 0):
                continue
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "") in TOOL_GEOMS:
                continue
            best = min(best, float(mujoco.mj_geomDistance(self.model, self.data, g, gid, distmax, fromto)))
        return best

    def ik(
        self,
        target_pos: np.ndarray,
        q0: np.ndarray,
        tool_axis_world: np.ndarray | None = np.array([0.0, 0.0, -1.0]),
        posture: np.ndarray = HOME,
        iters: int = 200,
        tol: float = 1e-4,
    ) -> tuple[np.ndarray, float]:
        """Damped least-squares IK for the tool tip position, with the tool axis (+z of the tool
        body) aligned to ``tool_axis_world`` as a secondary objective and a posture bias in the
        nullspace. Returns (q, final position error in m)."""
        q = q0.copy()
        lo, hi = self.joint_range[:, 0], self.joint_range[:, 1]
        err = np.inf
        for _ in range(iters):
            pos, rot = self.tool_frame(q)
            e_pos = target_pos - pos
            err = float(np.linalg.norm(e_pos))
            jac_p, jac_r = self.tool_jacobian(q)
            if tool_axis_world is None:
                jac, e = jac_p, e_pos
            else:
                axis = rot[:, 2]
                e_rot = np.cross(axis, tool_axis_world)  # small-angle rotation vector
                jac = np.vstack([jac_p, 0.3 * jac_r])
                e = np.concatenate([e_pos, 0.3 * e_rot])
            if err < tol and (tool_axis_world is None or np.linalg.norm(e_rot) < 1e-2):
                break
            lam = 1e-3
            jjt = jac @ jac.T + lam * np.eye(jac.shape[0])
            dq = jac.T @ np.linalg.solve(jjt, e)
            null = np.eye(NUM_JOINTS) - jac.T @ np.linalg.solve(jjt, jac)
            dq += 0.05 * (null @ (posture - q))
            step = np.clip(dq, -0.2, 0.2)
            q = np.clip(q + step, lo + 0.05, hi - 0.05)
        return q, err
