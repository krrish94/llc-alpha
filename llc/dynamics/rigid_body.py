"""Batched, differentiable rigid-body dynamics of a fixed-base arm, built from a MuJoCo ``MjModel``.

The model is assembled from compiled ``MjModel`` fields (never by re-parsing XML) and reproduces
MuJoCo's free-space quantities: forward kinematics, site Jacobian (``mj_jacSite``), joint-space
inertia (``mj_fullM``), bias forces (``qfrc_bias``), inverse dynamics (``mj_inverse``) and forward
dynamics (``qacc``).  All functions are batched over a leading batch dimension and written with
3-vector / 3x3 math, so the only Python loop is over the ~10 bodies, never over the batch.  Everything
is plain torch, so autograd works with respect to ``q``, ``qd``, ``tau`` and the inertial parameters
(``mass``, ``com``, ``inertia``, ``armature``, ``damping``, ``gravity``; call ``.requires_grad_()``).

MuJoCo conventions used here (see ``mj_kinematics``):

* Body frame: ``R_i = R_parent R(body_quat_i) R(jnt_axis_i, q_i)``.  A hinge rotates the body about an
  axis through the joint anchor ``jnt_pos_i`` (given in the body frame).  Internally every body is
  tracked at its joint anchor, a point fixed in both the body and its parent, so all offsets are
  constants; the MuJoCo body origin is ``anchor_i - R_i jnt_pos_i``.  Bodies without a joint (link0,
  tool) are welded to their parent, with the anchor at the body origin.
* ``body_inertia`` is the diagonal inertia in the inertial frame (``body_ipos``, ``body_iquat``).  The
  inertia about the COM expressed in the body frame is ``R(iquat) diag(inertia) R(iquat)^T``.
* The joint-space inertia includes ``dof_armature`` on the diagonal.  ``bias`` is Coriolis/centrifugal
  plus gravity without damping (``qfrc_bias``); ``passive = -dof_damping * qd`` (``qfrc_passive``);
  ``mj_inverse`` returns ``M qdd + bias - passive``.  ``dof_frictionloss`` is stored but not modelled
  (it is zero for the Panda scene).
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
import torch

_cross = torch.linalg.cross  # broadcasting 3-vector cross product


def _mv(R, v):
    """Batched matrix-vector product with broadcasting: (..., 3, 3) @ (..., 3) -> (..., 3)."""
    return (R @ v.unsqueeze(-1)).squeeze(-1)


def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def _skew(a):
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


class RigidBodyModel:
    """Rigid-body model of a kinematic tree of hinge joints (0 or 1 hinge per body), from an ``MjModel``.

    Inputs ``q``, ``qd``, ``qdd``, ``tau`` are ``(B, nv)`` tensors of the model's dtype/device; ``f_tool``
    is a world-frame force ``(B, 3)`` applied at the ``tool_tip`` site.  Body-indexed outputs use MuJoCo
    body ids (index 0 is the world body).
    """

    def __init__(self, model, dtype=torch.float64, device="cpu", tool_site="tool_tip"):
        if not isinstance(model, mujoco.MjModel):
            model = mujoco.MjModel.from_xml_path(os.fspath(model))
        self.dtype = dtype
        self.device = torch.device(device)

        def tensor(x):
            return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype, device=self.device)

        self.nbody, self.nv = int(model.nbody), int(model.nv)
        self.parent = [int(p) for p in model.body_parentid]
        self.dof = [-1] * self.nbody  # dof index of each body's hinge, -1 if welded
        # Per-body constants indexed by MuJoCo body id (entry 0 = world is unused).
        self._R_off = [None]  # rotation of the body frame (q=0) relative to the parent frame
        self._r = [None]  # vector from the parent's anchor to this body's anchor, in the parent frame
        self._axis, self._K, self._K2 = [None], [None], [None]  # hinge axis (body frame), skew(axis), skew^2
        jnt_pos = np.zeros((self.nbody, 3))  # anchor in the body's own frame (0 for welded bodies)
        for b in range(1, self.nbody):
            if model.body_jntnum[b] == 1:
                j = int(model.body_jntadr[b])
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError(f"body {b}: only hinge joints are supported")
                self.dof[b] = int(model.jnt_dofadr[j])
                jnt_pos[b] = model.jnt_pos[j]
                K = _skew(model.jnt_axis[j])
                self._axis.append(tensor(model.jnt_axis[j]))
                self._K.append(tensor(K))
                self._K2.append(tensor(K @ K))
            elif model.body_jntnum[b] == 0:
                self._axis.append(None)
                self._K.append(None)
                self._K2.append(None)
            else:
                raise ValueError(f"body {b}: at most one joint per body is supported")
            R_off = _quat_to_mat(model.body_quat[b])
            self._R_off.append(tensor(R_off))
            self._r.append(tensor(model.body_pos[b] + R_off @ jnt_pos[b] - jnt_pos[self.parent[b]]))
        self._jnt_pos = tensor(jnt_pos)
        self._joint_bodies = sorted((b for b in range(self.nbody) if self.dof[b] >= 0), key=self.dof.__getitem__)

        # Inertial / joint parameters, per MuJoCo body id (world has zero mass) and per dof.
        self.mass = tensor(model.body_mass)  # (nbody,)
        self.com = tensor(model.body_ipos)  # (nbody, 3) COM in the body frame
        R_i = [_quat_to_mat(model.body_iquat[b]) for b in range(self.nbody)]
        self.inertia = tensor([R_i[b] @ np.diag(model.body_inertia[b]) @ R_i[b].T for b in range(self.nbody)])
        self.armature = tensor(model.dof_armature)  # (nv,)
        self.damping = tensor(model.dof_damping)  # (nv,)
        self.frictionloss = tensor(model.dof_frictionloss)  # (nv,) stored only
        self.gravity = tensor(model.opt.gravity)  # (3,)

        # Tool site: position/orientation relative to its body's anchor.
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tool_site)
        if sid < 0:
            raise ValueError(f"site {tool_site!r} not found")
        self.tool_body = int(model.site_bodyid[sid])
        self._site_pos = tensor(model.site_pos[sid] - jnt_pos[self.tool_body])
        self._site_rot = tensor(_quat_to_mat(model.site_quat[sid]))
        self._tool_chain = []  # jointed bodies on the path from the tool body to the root
        b = self.tool_body
        while b > 0:
            if self.dof[b] >= 0:
                self._tool_chain.append(b)
            b = self.parent[b]

    # ----------------------------------------------------------------------------- kinematics
    def _kinematics(self, q):
        """World rotation, anchor position, hinge axis and anchor offset of every body (lists by body id)."""
        B = q.shape[0]
        sin, versin = q.sin().unbind(-1), (1.0 - q.cos()).unbind(-1)
        eye = torch.eye(3, dtype=q.dtype, device=q.device).expand(B, 3, 3)
        R, anchor, axis, d = [eye], [q.new_zeros(B, 3)], [None], [None]
        for b in range(1, self.nbody):
            p = self.parent[b]
            R_pre = R[p] @ self._R_off[b]
            d.append(_mv(R[p], self._r[b]))
            anchor.append(anchor[p] + d[b])
            j = self.dof[b]
            if j >= 0:  # Rodrigues: R(axis, q) = I + sin(q) K + (1 - cos(q)) K^2
                R_j = eye + sin[j].view(B, 1, 1) * self._K[b] + versin[j].view(B, 1, 1) * self._K2[b]
                R.append(R_pre @ R_j)
                axis.append(_mv(R_pre, self._axis[b]))
            else:
                R.append(R_pre)
                axis.append(None)
        return R, anchor, axis, d

    def _tool_pos(self, kin):
        R, anchor, _, _ = kin
        return anchor[self.tool_body] + _mv(R[self.tool_body], self._site_pos)

    def _tool_jacobian(self, kin, p_tool):
        """Positional Jacobian (B, 3, nv) of the tool site: column j = axis_j x (p_tool - anchor_j)."""
        _, anchor, axis, _ = kin
        cols = [p_tool.new_zeros(p_tool.shape)] * self.nv
        for b in self._tool_chain:
            cols[self.dof[b]] = _cross(axis[b], p_tool - anchor[b])
        return torch.stack(cols, -1)

    def fk(self, q):
        """Forward kinematics: body origins ``pos`` (B, nbody, 3), rotations ``rot`` (B, nbody, 3, 3), and the
        tool site's ``tool_pos`` (B, 3) and ``tool_rot`` (B, 3, 3), all in the world frame."""
        kin = self._kinematics(q)
        R, anchor, _, _ = kin
        pos = torch.stack([anchor[b] - _mv(R[b], self._jnt_pos[b]) for b in range(self.nbody)], 1)
        return {
            "pos": pos,
            "rot": torch.stack(R, 1),
            "tool_pos": self._tool_pos(kin),
            "tool_rot": R[self.tool_body] @ self._site_rot,
        }

    def tool_pos(self, q):
        """World position (B, 3) of the tool site."""
        return self._tool_pos(self._kinematics(q))

    def tool_jacobian(self, q):
        """Positional Jacobian (B, 3, nv) of the tool site, matching ``mj_jacSite``."""
        kin = self._kinematics(q)
        return self._tool_jacobian(kin, self._tool_pos(kin))

    # ----------------------------------------------------------------------------- dynamics
    def _rnea(self, kin, qd, qdd, a0):
        """Recursive Newton-Euler in world coordinates, batched over (B, K).

        ``qd``, ``qdd``: (B, K, nv); ``a0``: fictitious base acceleration broadcastable to (B, K, 3)
        (``-gravity`` to include gravity, ``0`` to exclude it).  Returns joint torques (B, K, nv).
        The K axis lets several (qd, qdd, gravity) problems share one kinematics evaluation.
        """
        R, anchor, axis, d = kin
        mass, inertia = self.mass.unbind(0), self.inertia.unbind(0)
        com = (self.com - self._jnt_pos).unbind(0)  # COM relative to the anchor, body frame
        armature = self.armature.unbind(0)
        qd, qdd = qd.unsqueeze(-1).unbind(-2), qdd.unsqueeze(-1).unbind(-2)  # per dof: (B, K, 1)
        zero = qd[0].new_zeros(1, 1, 3)
        w, al, a = [zero], [zero], [a0]  # angular velocity / acceleration, anchor acceleration
        f, n = [None] * self.nbody, [None] * self.nbody  # force and torque (about the anchor) on each subtree
        for b in range(1, self.nbody):
            p = self.parent[b]
            d_b = d[b].unsqueeze(1)
            if p == 0:  # the world does not move
                w_b, al_b, a_b = zero, zero, a0
            else:  # the anchor is fixed in the parent
                w_b, al_b = w[p], al[p]
                a_b = a[p] + _cross(al_b, d_b) + _cross(w_b, _cross(w_b, d_b))
            j = self.dof[b]
            if j >= 0:
                s_b = axis[b].unsqueeze(1)
                al_b = al_b + s_b * qdd[j] + _cross(w_b, s_b) * qd[j]
                w_b = w_b + s_b * qd[j]
            w.append(w_b)
            al.append(al_b)
            a.append(a_b)
            # Newton-Euler about the COM, then referred to the anchor.
            c_b = _mv(R[b], com[b]).unsqueeze(1)
            a_c = a_b + _cross(al_b, c_b) + _cross(w_b, _cross(w_b, c_b))
            F = mass[b] * a_c
            I_w = (R[b] @ inertia[b] @ R[b].transpose(-1, -2)).unsqueeze(1)
            N = _mv(I_w, al_b) + _cross(w_b, _mv(I_w, w_b))
            f[b], n[b] = F, N + _cross(c_b, F)
        tau = [None] * self.nv
        for b in range(self.nbody - 1, 0, -1):
            j, p = self.dof[b], self.parent[b]
            if j >= 0:
                tau[j] = (axis[b].unsqueeze(1) * n[b]).sum(-1) + armature[j] * qdd[j].squeeze(-1)
            if p > 0:
                f[p] = f[p] + f[b]
                n[p] = n[p] + n[b] + _cross(d[b].unsqueeze(1), f[b])
        return torch.stack(tau, -1)

    def _mass_and_bias(self, kin, qd):
        """M(q) (B, nv, nv) and bias(q, qd) (B, nv) from one RNEA pass with K = 1 + nv stacked problems:
        problem 0 is (qd, qdd=0, gravity on); problems 1..nv are (qd=0, qdd=e_j, gravity off)."""
        B, nv = qd.shape
        eye = torch.eye(nv, dtype=qd.dtype, device=qd.device)
        qd_s = torch.cat([qd.unsqueeze(1), qd.new_zeros(B, nv, nv)], 1)
        qdd_s = torch.cat([qd.new_zeros(B, 1, nv), eye.expand(B, nv, nv)], 1)
        a0 = torch.cat([-self.gravity.view(1, 1, 3), qd.new_zeros(1, nv, 3)], 1)
        tau = self._rnea(kin, qd_s, qdd_s, a0)
        return tau[:, 1:].transpose(1, 2), tau[:, 0]

    def mass_matrix(self, q):
        """Joint-space inertia (B, nv, nv) including armature, matching ``mj_fullM``."""
        B, nv = q.shape
        kin = self._kinematics(q)
        qdd = torch.eye(nv, dtype=q.dtype, device=q.device).expand(B, nv, nv)
        return self._rnea(kin, q.new_zeros(B, nv, nv), qdd, q.new_zeros(1, 1, 3)).transpose(1, 2)

    def bias(self, q, qd):
        """Coriolis/centrifugal + gravity torques (B, nv), matching ``qfrc_bias`` (no damping)."""
        return self.inverse_dynamics(q, qd, torch.zeros_like(qd))

    def inverse_dynamics(self, q, qd, qdd):
        """M(q) qdd + bias(q, qd) (B, nv), including armature * qdd; equals ``qfrc_inverse + passive(qd)``."""
        kin = self._kinematics(q)
        a0 = -self.gravity.view(1, 1, 3)
        return self._rnea(kin, qd.unsqueeze(1), qdd.unsqueeze(1), a0).squeeze(1)

    def passive(self, qd):
        """Passive joint torques -damping * qd (B, nv), matching ``qfrc_passive`` (frictionloss = 0)."""
        return -self.damping * qd

    def _generalized_force(self, kin, tau, bias, f_tool):
        rhs = tau - bias
        if f_tool is not None:
            J = self._tool_jacobian(kin, self._tool_pos(kin))
            rhs = rhs + _mv(J.transpose(1, 2), f_tool)
        return rhs

    def forward_dynamics(self, q, qd, tau, f_tool=None):
        """qdd = M^{-1} (tau + passive(qd) + J^T f_tool - bias) (B, nv), matching ``qacc`` in free space."""
        kin = self._kinematics(q)
        M, bias = self._mass_and_bias(kin, qd)
        rhs = self._generalized_force(kin, tau + self.passive(qd), bias, f_tool)
        return torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)

    def step(self, q, qd, tau, dt, f_tool=None):
        """One semi-implicit Euler step with implicit viscous damping:

            (M + dt D) qd_next = M qd + dt (tau + J^T f_tool - bias),   q_next = q + dt qd_next,

        with D = diag(damping).  This is equivalent to solving (M + dt D) qdd = tau + passive - bias and
        integrating velocity then position, which is exactly what MuJoCo's ``implicitfast`` integrator
        does for a model whose only velocity-dependent passive force is linear damping.  Free-space
        rollouts therefore agree with ``mj_step`` to roundoff; any constraints (joint limits, contacts)
        handled by MuJoCo's solver are not modelled here.
        """
        kin = self._kinematics(q)
        M, bias = self._mass_and_bias(kin, qd)
        rhs = _mv(M, qd) + dt * self._generalized_force(kin, tau, bias, f_tool)
        A = M + dt * torch.diag(self.damping)
        qd_next = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
        return q + dt * qd_next, qd_next

    # ----------------------------------------------------------------------------- tool and contact
    def tool_state(self, q, qd):
        """World position and velocity (B, 3) of the tool site from one kinematics evaluation."""
        kin = self._kinematics(q)
        p = self._tool_pos(kin)
        return p, _mv(self._tool_jacobian(kin, p), qd)

    def step_contact(self, q, qd, tau, dt, contact_params, geometry=None):
        """``step`` with the penalty contact force of ``llc.dynamics.contact`` applied at the tool site,
        using a single kinematics evaluation. Returns (q_next, qd_next, f_tool, tool_pos)."""
        from llc.dynamics.contact import contact_force

        kin = self._kinematics(q)
        p = self._tool_pos(kin)
        v = _mv(self._tool_jacobian(kin, p), qd)
        f_tool, _ = contact_force(p, v, contact_params, geometry)
        M, bias = self._mass_and_bias(kin, qd)
        rhs = _mv(M, qd) + dt * self._generalized_force(kin, tau, bias, f_tool)
        A = M + dt * torch.diag(self.damping)
        qd_next = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
        return q + dt * qd_next, qd_next, f_tool, p
