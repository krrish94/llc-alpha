"""Differentiable penalty contact between the tool-tip sphere and the static scene geometry.

The scene has three static surfaces the tool can touch, all parsed from the ``MjModel`` (never hard
coded): the floor (plane z = 0), the top face of the table box and the robot-facing (-x) face of the
wall box.  Each is treated as a planar face with normal ``n`` and, for the boxes, lateral extents.

For a sphere of radius ``r`` with centre ``p`` and velocity ``v`` the model is, per face,

    d   = max(0, r - (n . p - offset))                     penetration depth
    w   = smooth indicator that p projects onto the face   (1 inside, 0 one radius outside the edge)
    f_n = w * max(0, k_n d - c_n v_n)    if d > 0, else 0  spring-damper normal force along n
    f_t = -mu f_n v_t / sqrt(|v_t|^2 + eps^2)              regularised Coulomb friction

with ``v_n = n . v`` and ``v_t = v - v_n n``.  The simple spring-damper normal law is used rather than a
Hunt-Crossley form (``k_n d (1 + c_n (-v_n))``); it is linear in the state but, like MuJoCo's
contacts, can jump at impact when ``v_n < 0``.  The lateral weight makes a sphere that passes a
box edge lose contact smoothly instead of through a step.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

_DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "sim" / "assets" / "panda_scene.xml"


@dataclass
class ContactParams:
    """Penalty contact parameters.

    The defaults approximate MuJoCo's ``solref="0.005 1"`` contacts (5 ms time constant, critically
    damped) for the ~1 kg effective mass at the tool tip; they are to be calibrated separately against
    the simulator.
    """

    k_n: float = 5000.0  # normal stiffness [N/m]
    c_n: float = 50.0  # normal damping [N s/m]
    mu: float = 0.8  # Coulomb friction coefficient
    eps: float = 1e-3  # regularisation velocity of the Coulomb law [m/s]


@dataclass(frozen=True)
class Face:
    """Planar face {p : normal . p = offset}, bounded along ``bounds = ((axis, lo, hi), ...)``."""

    normal: tuple[float, float, float]
    offset: float
    bounds: tuple[tuple[int, float, float], ...] = ()


@dataclass(frozen=True)
class SceneGeometry:
    """Tool-tip radius and the faces (floor, table top, wall front) parsed from an ``MjModel``."""

    radius: float
    faces: tuple[Face, ...]

    @classmethod
    def from_model(cls, model) -> "SceneGeometry":
        if not isinstance(model, mujoco.MjModel):
            model = mujoco.MjModel.from_xml_path(os.fspath(model))

        def geom(name):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(f"geom {name!r} not found")
            if not np.allclose(model.geom_quat[gid], [1, 0, 0, 0]):
                raise ValueError(f"geom {name!r} must be axis aligned")
            return model.geom_pos[gid].astype(float), model.geom_size[gid].astype(float)

        radius = geom("tool_tip")[1][0]
        floor_pos, _ = geom("floor")
        table_pos, table_size = geom("table")
        wall_pos, wall_size = geom("wall")
        faces = (
            Face((0.0, 0.0, 1.0), floor_pos[2]),
            Face(
                (0.0, 0.0, 1.0),
                table_pos[2] + table_size[2],
                (
                    (0, table_pos[0] - table_size[0], table_pos[0] + table_size[0]),
                    (1, table_pos[1] - table_size[1], table_pos[1] + table_size[1]),
                ),
            ),
            Face(
                (-1.0, 0.0, 0.0),
                -(wall_pos[0] - wall_size[0]),
                (
                    (1, wall_pos[1] - wall_size[1], wall_pos[1] + wall_size[1]),
                    (2, wall_pos[2] - wall_size[2], wall_pos[2] + wall_size[2]),
                ),
            ),
        )
        return cls(radius, faces)


@functools.lru_cache(maxsize=None)
def default_geometry() -> SceneGeometry:
    return SceneGeometry.from_model(_DEFAULT_SCENE)


def _smoothstep(x):
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _face_force(p, v, face, r, params):
    n = p.new_tensor(face.normal)
    d = torch.relu(r - (p @ n - face.offset))
    w = torch.ones_like(d)
    for axis, lo, hi in face.bounds:
        w = w * _smoothstep((p[:, axis] - lo + r) / r) * _smoothstep((hi - p[:, axis] + r) / r)
    v_n = v @ n
    f_n = torch.relu(params.k_n * d - params.c_n * v_n) * w
    f_n = torch.where(d > 0, f_n, torch.zeros_like(f_n))
    v_t = v - v_n.unsqueeze(-1) * n
    f_t = -params.mu * f_n.unsqueeze(-1) * v_t / torch.sqrt((v_t * v_t).sum(-1, keepdim=True) + params.eps**2)
    return f_n.unsqueeze(-1) * n + f_t, d * w


def contact_force(tool_pos, tool_vel, params: ContactParams, geometry: SceneGeometry | None = None):
    """World-frame contact force (B, 3) on the tool sphere and effective penetration (B, 3) per face.

    ``tool_pos`` / ``tool_vel`` are the world position and velocity (B, 3) of the tool-tip site (the
    sphere centre).  Faces are ordered (floor, table top, wall front).
    """
    geometry = default_geometry() if geometry is None else geometry
    forces, depths = zip(*(_face_force(tool_pos, tool_vel, f, geometry.radius, params) for f in geometry.faces))
    return sum(forces), torch.stack(depths, -1)
