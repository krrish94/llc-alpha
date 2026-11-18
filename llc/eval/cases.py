"""Evaluation cases: a reference, an initial state, optional disturbance and plant perturbation.

Case sets are generated deterministically from (set name, regime, index) so that every
controller is evaluated on identical cases and paired comparisons are valid. The split names
'tune', 'val' and 'test' use disjoint seed ranges.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from llc import references as R
from llc.sim.nominal import NominalModel
from llc.sim.panda import NUM_JOINTS, DynamicsPerturbation
from llc.sim.plant import PLANT

REGIMES = ("hold", "smooth", "zoh", "step", "contact", "transition")
SPLIT_OFFSET = {"tune": 0, "val": 1000, "test": 2000, "train": 10000}


@dataclass
class Pulse:
    """External force pulse at the tool (world frame) used to measure compliance."""

    t_start: float
    duration: float
    force: tuple[float, float, float]


@dataclass
class Case:
    id: str
    regime: str
    seed: int
    reference: R.Reference
    q0: np.ndarray
    qd0: np.ndarray
    pulse: Pulse | None = None
    perturbation: DynamicsPerturbation = field(default_factory=DynamicsPerturbation)
    meta: dict = field(default_factory=dict)


def case_seed(split: str, regime: str, index: int) -> int:
    h = hashlib.sha256(f"{regime}-{index}".encode()).digest()
    return SPLIT_OFFSET[split] * 100000 + int.from_bytes(h[:3], "little") % 100000


def make_case(regime: str, split: str, index: int, nominal: NominalModel,
              perturbation: DynamicsPerturbation | None = None, speed_scale: float = 1.0) -> Case:
    """Build one case. ``speed_scale`` multiplies the sampled speed (used for out-of-distribution
    tests); the sampled nominal speed ranges are the in-distribution protocol."""
    seed = case_seed(split, regime, index)
    rng = np.random.default_rng(seed)
    pulse = None
    meta = {}
    if regime == "hold":
        ref = R.hold(rng, nominal)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        pulse = Pulse(1.0, 0.2, tuple(PLANT.pulse_force * direction))
        meta["pulse_force_N"] = 10.0
    elif regime == "smooth":
        speed = rng.uniform(0.2, 0.6) * speed_scale
        ref = R.smooth(rng, nominal, speed)
    elif regime == "zoh":
        speed = rng.uniform(0.2, 0.6) * speed_scale
        ref = R.zoh(rng, nominal, speed)
    elif regime == "step":
        ref = R.step(rng, nominal, step_scale=0.25 * speed_scale)
    elif regime == "contact":
        ref = R.contact(rng, nominal, slide_speed=rng.uniform(0.05, 0.15) * speed_scale)
    elif regime == "transition":
        ref = R.transition(rng, nominal, approach_speed=rng.uniform(0.05, 0.25) * speed_scale)
    else:
        raise ValueError(regime)
    meta.update({k: v for k, v in ref.meta.items() if k not in ("normal", "tool_path")})
    tool_ref = ref.meta.get("tool_path")
    if tool_ref is None:  # commanded tool path from forward kinematics of the joint reference
        tool_ref = np.array([nominal.tool_pos(ref.q[i]) for i in range(0, len(ref.q), 10)])
        tool_ref = np.array([np.interp(ref.t, ref.t[::10], tool_ref[:, j]) for j in range(3)]).T
    meta["tool_ref"] = tool_ref
    return Case(f"{regime}-{split}-{index:03d}", regime, seed, ref, ref.q[0].copy(), np.zeros(NUM_JOINTS), pulse,
                perturbation or DynamicsPerturbation(), meta)


def case_set(split: str, n_per_regime: int, nominal: NominalModel, regimes=REGIMES, **kw) -> list[Case]:
    return [make_case(r, split, i, nominal, **kw) for r in regimes for i in range(n_per_regime)]
