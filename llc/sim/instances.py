"""Robot instances: small, random deviations of the simulated robot from the nominal model, standing
in for the imperfect identification of a real Franka (deep-dive protocol, ADR 0005).

Evaluation uses a fixed set of instances (seeded); training samples a fresh instance per window.
The nominal model used by every controller for gravity compensation does not change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llc.sim.panda import DynamicsPerturbation


@dataclass
class InstanceRanges:
    mass_rel: float = 0.05  # +-5% per link
    inertia_rel: float = 0.10  # +-10% per link (log-uniform)
    com_offset_m: float = 0.005  # +-5 mm per axis
    damping_range: tuple[float, float] = (0.5, 2.0)  # log-uniform scale per joint
    armature_rel: float = 0.2  # +-20% per joint
    friction_loss_max: float = 0.3  # Nm, uniform per instance (same on all joints)
    torque_rel: float = 0.03  # +-3% per actuator
    obs_noise_q: float = 1e-4  # rad, nominal plant sensor noise (velocity noise 10x)


def sample_instance(rng: np.random.Generator, ranges: InstanceRanges = InstanceRanges(),
                    payload_mass: float = 0.0, payload_offset=(0.0, 0.0, 0.10)) -> DynamicsPerturbation:
    r = ranges
    u = lambda lo, hi, n=None: rng.uniform(lo, hi, n)
    return DynamicsPerturbation(
        payload_mass=payload_mass, payload_offset=tuple(payload_offset),
        friction_loss=float(u(0.0, r.friction_loss_max)),
        obs_noise_q=r.obs_noise_q, obs_noise_qd=10.0 * r.obs_noise_q,
        link_mass_scale=tuple(u(1 - r.mass_rel, 1 + r.mass_rel, 8).tolist()),
        link_inertia_scale=tuple(np.exp(u(np.log(1 - r.inertia_rel), np.log(1 + r.inertia_rel), 8)).tolist()),
        link_com_offset=tuple(tuple(x) for x in u(-r.com_offset_m, r.com_offset_m, (8, 3)).tolist()),
        joint_damping_scale=tuple(np.exp(u(np.log(r.damping_range[0]), np.log(r.damping_range[1]), 7)).tolist()),
        joint_armature_scale=tuple(u(1 - r.armature_rel, 1 + r.armature_rel, 7).tolist()),
        torque_scale=tuple(u(1 - r.torque_rel, 1 + r.torque_rel, 7).tolist()),
    )


def evaluation_instances(n: int, seed: int = 2026, **kw) -> list[DynamicsPerturbation]:
    """Fixed, seeded instances for evaluation; instance 0 is the nominal robot with nominal noise."""
    rng = np.random.default_rng(seed)
    nominal = DynamicsPerturbation(obs_noise_q=InstanceRanges().obs_noise_q, obs_noise_qd=10 * InstanceRanges().obs_noise_q, **kw)
    return [nominal] + [sample_instance(rng, **kw) for _ in range(n - 1)]
