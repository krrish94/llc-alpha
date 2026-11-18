"""Learned robot dynamics (Milestone 4): a network predicting the joint acceleration (and the
tool contact force magnitude) from (q, qd, tau), integrated with the same semi-implicit Euler
step as the known model. Two variants:

- 'blackbox': qdd = f(q, qd, tau)
- 'residual': qdd = qdd_known(q, qd, tau) + f(q, qd, tau) with the known free-space rigid-body model.

The controller structure keeps gravity compensation from the known nominal model, as an engineer
would, so ``gravity`` is not learned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from llc.dynamics.rigid_body import RigidBodyModel
from llc.learning.rollout import DynamicsInterface
from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS, load_model


@dataclass
class LearnedModelConfig:
    variant: str = "blackbox"  # or "residual"
    hidden: int = 256
    layers: int = 3
    dt: float = 0.0025
    # normalization (set from data)
    q_mean: tuple = (0.0,) * NUM_JOINTS
    q_std: tuple = (1.0,) * NUM_JOINTS
    qd_std: tuple = (1.0,) * NUM_JOINTS
    qdd_std: tuple = (1.0,) * NUM_JOINTS
    f_scale: float = 50.0


class LearnedDynamicsNet(nn.Module):
    def __init__(self, cfg: LearnedModelConfig):
        super().__init__()
        self.cfg = cfg
        dims = [21] + [cfg.hidden] * cfg.layers
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.head_qdd = nn.Linear(cfg.hidden, NUM_JOINTS)
        self.head_f = nn.Linear(cfg.hidden, 1)
        nn.init.zeros_(self.head_qdd.weight); nn.init.zeros_(self.head_qdd.bias)
        for name in ("q_mean", "q_std", "qd_std", "qdd_std"):
            self.register_buffer(name, torch.tensor(getattr(cfg, name), dtype=torch.float32))
        self.register_buffer("limits", torch.tensor(TORQUE_LIMITS, dtype=torch.float32))

    def forward(self, q, qd, tau):
        x = torch.cat([(q - self.q_mean) / self.q_std, qd / self.qd_std, tau / self.limits], -1)
        h = self.trunk(x)
        return self.head_qdd(h) * self.qdd_std, torch.nn.functional.softplus(self.head_f(h)).squeeze(-1) * self.cfg.f_scale


class LearnedDynamics(DynamicsInterface):
    """Dynamics interface for training controllers through the learned model."""

    def __init__(self, net: LearnedDynamicsNet, known: RigidBodyModel):
        self.net, self.known = net, known
        self.variant = net.cfg.variant

    def gravity(self, q):
        return self.known.bias(q, torch.zeros_like(q))

    def tool_jacobian(self, q):
        return self.known.tool_jacobian(q)

    def acceleration(self, q, qd, tau):
        qdd, f_mag = self.net(q, qd, tau)
        if self.variant == "residual":
            qdd = qdd + self.known.forward_dynamics(q, qd, tau)
        return qdd, f_mag

    def step(self, q, qd, tau, dt):
        qdd, f_mag = self.acceleration(q, qd, tau)
        qd_next = qd + dt * qdd
        q_next = q + dt * qd_next
        p = self.known.tool_pos(q)
        f = torch.stack([torch.zeros_like(f_mag), torch.zeros_like(f_mag), f_mag], -1)  # magnitude only; direction unused by the loss
        return q_next, qd_next, f, p


def save_model(path, net: LearnedDynamicsNet, extra: dict | None = None) -> None:
    torch.save({"config": asdict(net.cfg), "state_dict": net.state_dict(), "extra": extra or {}}, path)


def load_model_net(path) -> tuple[LearnedDynamicsNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = LearnedModelConfig(**ckpt["config"])
    net = LearnedDynamicsNet(cfg)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt.get("extra", {})


def load_learned_dynamics(path) -> LearnedDynamics:
    net, _ = load_model_net(path)
    return LearnedDynamics(net, RigidBodyModel(load_model(), dtype=torch.float32))
