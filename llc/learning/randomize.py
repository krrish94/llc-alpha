"""Dynamics randomization for training through the torch model (Milestone 3).

The same quantities as ``DynamicsPerturbation`` of the simulator are varied: payload mass and
offset on the tool, a scale on all link inertias, joint dry friction, viscous damping scale,
actuation (compute) delay and observation noise. Inertial parameters are sampled once per
training iteration (the batch shares a plant); noise is sampled per sample and step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from llc.dynamics.rigid_body import RigidBodyModel


@dataclass
class RandomizationRanges:
    payload_mass: tuple[float, float] = (0.0, 2.0)  # kg
    payload_lateral: tuple[float, float] = (0.0, 0.06)  # m, offset perpendicular to the tool axis
    inertia_scale: tuple[float, float] = (0.8, 1.3)
    friction_loss: tuple[float, float] = (0.0, 1.0)  # Nm
    damping_scale: tuple[float, float] = (0.7, 1.5)
    latency_s: tuple[float, float] = (0.0025, 0.0125)  # s, rounded to model steps
    obs_noise_q: tuple[float, float] = (0.0, 0.002)  # rad (velocity noise is 10x)


class ModelRandomizer:
    """Applies sampled perturbations to a RigidBodyModel in place and restores the nominal values."""

    def __init__(self, model: RigidBodyModel, ranges: RandomizationRanges):
        self.model, self.ranges = model, ranges
        self.nominal = {k: getattr(model, k).clone() for k in ("mass", "com", "inertia", "damping", "frictionloss")}

    def restore(self) -> None:
        for k, v in self.nominal.items():
            getattr(self.model, k).copy_(v)

    def sample(self, rng: np.random.Generator) -> dict:
        r = self.ranges
        vals = {"payload_mass": float(rng.uniform(*r.payload_mass)), "payload_lateral": float(rng.uniform(*r.payload_lateral)),
                "inertia_scale": float(rng.uniform(*r.inertia_scale)), "friction_loss": float(rng.uniform(*r.friction_loss)),
                "damping_scale": float(rng.uniform(*r.damping_scale)), "latency_s": float(rng.uniform(*r.latency_s)),
                "obs_noise_q": float(rng.uniform(*r.obs_noise_q))}
        self.apply(vals)
        return vals

    def apply(self, vals: dict) -> None:
        m = self.model
        self.restore()
        m.inertia.mul_(vals["inertia_scale"])
        m.damping.mul_(vals["damping_scale"])
        m.frictionloss.fill_(vals["friction_loss"])
        if vals["payload_mass"] > 0.0:
            b = m.tool_body
            m0, c0, I0 = m.mass[b].clone(), m.com[b].clone(), m.inertia[b].clone()
            mp = vals["payload_mass"]
            offset = torch.tensor([vals["payload_lateral"], 0.0, 0.10], dtype=m.dtype, device=m.device)
            m1 = m0 + mp
            c1 = (m0 * c0 + mp * offset) / m1
            eye = torch.eye(3, dtype=m.dtype, device=m.device)
            shift = lambda mass, d: mass * (torch.dot(d, d) * eye - torch.outer(d, d))
            m.mass[b] = m1
            m.com[b] = c1
            m.inertia[b] = I0 + shift(m0, c0 - c1) + shift(mp, offset - c1)
