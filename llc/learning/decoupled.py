"""Decoupled forward-backward gradients (gradient plan, item 2; after DMO, Amigo et al. 2025).

The plant (MuJoCo) produces every state in the forward pass; a learned one-step dynamics model,
fit online to the plant's own transitions, supplies the Jacobians in the backward pass through a
straight-through combination: value from the simulator, gradient from the model. No finite
differences are needed during training. The model is conditioned on the robot-instance context
(payload and parameter deviations) like SuperMimic's simulator context.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS, DynamicsPerturbation

CONTEXT_DIM = 34


def context_vector(p: DynamicsPerturbation) -> np.ndarray:
    """Instance context: payload mass and offset, per-link mass and inertia scales, joint damping
    scales, dry friction and actuator gain scales (34 numbers, centred so the nominal robot is 0)."""
    return np.concatenate([
        [p.payload_mass / 3.0], np.asarray(p.payload_offset[:2]) / 0.03,
        np.asarray(p.link_mass_scale) - 1.0, np.asarray(p.link_inertia_scale) - 1.0,
        np.log(np.asarray(p.joint_damping_scale)), [p.friction_loss / 0.3], np.asarray(p.torque_scale) - 1.0,
    ]).astype(np.float32)


class DynamicsNet(nn.Module):
    """(q, qd, tau, context) -> (velocity increment over one step, touch force). Inputs use fixed
    scales; the increment is normalised by a running estimate of its standard deviation."""

    def __init__(self, hidden: int = 512, layers: int = 3):
        super().__init__()
        dims = [3 * NUM_JOINTS + CONTEXT_DIM] + [hidden] * layers
        mods: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU()]
        self.trunk = nn.Sequential(*mods)
        self.head = nn.Linear(hidden, NUM_JOINTS + 1)
        self.register_buffer("limits", torch.tensor(TORQUE_LIMITS, dtype=torch.float32))
        self.register_buffer("dqd_std", torch.full((NUM_JOINTS,), 0.05))

    def forward(self, q, qd, tau, xi):
        x = torch.cat([q / 3.0, qd / 2.0, tau / self.limits, xi], -1)
        out = self.head(self.trunk(x))
        return out[:, :NUM_JOINTS] * self.dqd_std, torch.nn.functional.softplus(out[:, NUM_JOINTS]) * 50.0


class Replay:
    def __init__(self, capacity: int):
        self.cap, self.n, self.i = capacity, 0, 0
        self.q = np.zeros((capacity, NUM_JOINTS), np.float32); self.qd = np.zeros_like(self.q); self.tau = np.zeros_like(self.q)
        self.xi = np.zeros((capacity, CONTEXT_DIM), np.float32); self.dqd = np.zeros_like(self.q); self.touch = np.zeros(capacity, np.float32)

    def add(self, q, qd, tau, xi, dqd, touch):
        b = len(q)
        idx = (self.i + np.arange(b)) % self.cap
        self.q[idx], self.qd[idx], self.tau[idx], self.xi[idx], self.dqd[idx], self.touch[idx] = q, qd, tau, xi, dqd, touch
        self.i = (self.i + b) % self.cap
        self.n = min(self.cap, self.n + b)

    def sample(self, k: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n, k)
        t = lambda a: torch.as_tensor(a[idx])
        return t(self.q), t(self.qd), t(self.tau), t(self.xi), t(self.dqd), t(self.touch)


class DecoupledBatch:
    """Drop-in replacement for MjBatch in the training rollouts: same reset/state/step/perturb
    interface; the simulator steps without finite differences, the model provides the gradients."""

    def __init__(self, mjbatch, net: DynamicsNet, rng: np.random.Generator, capacity: int = 2_000_000, dt: float = 1e-3, lr: float = 1e-3):
        self.sim, self.net, self.rng, self.dt = mjbatch, net, rng, dt
        self.replay = Replay(capacity)
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self.xi = torch.zeros(len(mjbatch.sims), CONTEXT_DIM)
        self.perturb(mjbatch.perturbations if hasattr(mjbatch, "perturbations") else [DynamicsPerturbation()] * len(mjbatch.sims))
        self.touch_grad_local = False
        self.record = True

    @property
    def sims(self):
        return self.sim.sims

    def perturb(self, perturbations) -> None:
        self.sim.perturb(perturbations)
        self.xi = torch.as_tensor(np.stack([context_vector(p) for p in perturbations]))

    def reset(self, q0, qd0=None) -> None:
        self.sim.reset(q0, qd0)

    def state(self):
        return self.sim.state()

    def step(self, q, qd, tau, f_ext=None):
        with torch.no_grad():
            q_s, qd_s, t_s, tool_pos = self.sim.step(q.detach(), qd.detach(), tau.detach(), f_ext)
            q_s, qd_s, t_s = q_s.float(), qd_s.float(), t_s.float()
        dqd_m, touch_m = self.net(q, qd, tau, self.xi)
        qd_m = qd + dqd_m
        q_m = q + self.dt * qd_m
        if self.record:
            self.replay.add(q.detach().numpy(), qd.detach().numpy(), tau.detach().numpy(), self.xi.numpy(), (qd_s - qd.detach()).numpy(), t_s.numpy())
        # straight-through: values from the simulator, gradients from the model
        q_n = q_m + (q_s - q_m).detach()
        qd_n = qd_m + (qd_s - qd_m).detach()
        touch_n = touch_m + (t_s - touch_m).detach()
        return q_n, qd_n, touch_n, tool_pos

    def fit(self, batches: int = 10, batch_size: int = 2048) -> dict:
        if self.replay.n < batch_size:
            return {"model_loss": float("nan")}
        with torch.no_grad():
            sample = self.replay.sample(min(self.replay.n, 65536), self.rng)
            self.net.dqd_std.copy_(0.9 * self.net.dqd_std + 0.1 * (sample[4].std(0) + 1e-4))
        last = {}
        for _ in range(batches):
            q, qd, tau, xi, dqd, touch = self.replay.sample(batch_size, self.rng)
            pred_dqd, pred_touch = self.net(q, qd, tau, xi)
            loss_dqd = ((pred_dqd - dqd) / self.net.dqd_std).square().mean()
            loss_touch = ((pred_touch - touch) / 50.0).square().mean()
            loss = loss_dqd + loss_touch
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            last = {"model_loss": loss.item(), "model_dqd": loss_dqd.item(), "model_touch": loss_touch.item()}
        return last


class ImaginedBatch:
    """Gradient plan, item 5: full imagination through the learned model (SuperMimic-style). The
    forward pass no longer touches the simulator; the model is kept fresh by separate plant
    rollouts recorded into the same replay (``collect`` on a DecoupledBatch). Same interface as
    MjBatch/DecoupledBatch so the actor-critic rollout runs unchanged."""

    def __init__(self, net: DynamicsNet, B: int, dt: float = 1e-3):
        self.net, self.B, self.dt = net, B, dt
        self.xi = torch.zeros(B, CONTEXT_DIM)
        self.q = torch.zeros(B, NUM_JOINTS); self.qd = torch.zeros(B, NUM_JOINTS)
        self.touch_grad_local = False

    def perturb(self, perturbations) -> None:
        self.xi = torch.as_tensor(np.stack([context_vector(p) for p in perturbations]))

    def reset(self, q0, qd0=None) -> None:
        self.q = torch.as_tensor(np.asarray(q0), dtype=torch.float32).clone()
        self.qd = torch.zeros_like(self.q) if qd0 is None else torch.as_tensor(np.asarray(qd0), dtype=torch.float32).clone()

    def state(self):
        return self.q.clone(), self.qd.clone()

    def step(self, q, qd, tau, f_ext=None):
        dqd_m, touch_m = self.net(q, qd, tau, self.xi)
        qd_n = qd + dqd_m
        q_n = q + self.dt * qd_n
        self.q, self.qd = q_n.detach(), qd_n.detach()
        return q_n, qd_n, touch_m, q_n.detach().numpy()[:, :3] * 0.0  # tool position not available in imagination
