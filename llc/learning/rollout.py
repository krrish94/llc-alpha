"""Differentiable closed-loop rollout of a policy through a torch dynamics model, with the
network rate, compute latency and history of ADR 0002, and the training loss.

The rollout is split into chunks for truncated backpropagation through time: the loss of each
chunk is backpropagated immediately and the state is detached at chunk boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from llc.dynamics.contact import ContactParams
from llc.learning.policy import Policy, encode_obs
from llc.sim.panda import NUM_JOINTS


@dataclass
class LossConfig:
    """Weighted sum of normalized penalties (scales in physical units)."""

    pos_scale: float = 0.02  # rad
    vel_weight: float = 0.1
    vel_scale: float = 0.5  # rad/s
    torque_weight: float = 0.02  # on (tau / limit)^2
    rate_weight: float = 0.03  # chosen so that the loss and the composite metric agree at 1000 Nm/s ~ 8 mrad RMSE
    rate_scale: float = 500.0  # Nm/s
    force_weight: float = 0.5
    force_soft: float = 40.0  # N, force above which contact force is penalized
    force_scale: float = 20.0  # N
    impulse_weight: float = 0.02
    impulse_scale: float = 2000.0  # N/s, rate of change of contact force


@dataclass
class RolloutConfig:
    dt: float = 0.0025  # model step
    horizon_s: float = 3.0
    tbptt_steps: int = 400
    contact: ContactParams = field(default_factory=ContactParams)

    @property
    def horizon_steps(self) -> int:
        return int(round(self.horizon_s / self.dt))


class DynamicsInterface:
    """What the rollout needs from a dynamics model: gravity torque, the tool Jacobian (to apply
    external forces) and a contact step."""

    def gravity(self, q):  # (B,7) -> (B,7)
        raise NotImplementedError

    def tool_jacobian(self, q):  # (B,7) -> (B,3,7)
        raise NotImplementedError

    def step(self, q, qd, tau, dt):  # -> (q_next, qd_next, f_tool (B,3), tool_pos (B,3))
        raise NotImplementedError


class RigidBodyDynamics(DynamicsInterface):
    def __init__(self, model, contact: ContactParams):
        self.model, self.contact = model, contact

    def gravity(self, q):
        return self.model.bias(q, torch.zeros_like(q))

    def tool_jacobian(self, q):
        return self.model.tool_jacobian(q)

    def step(self, q, qd, tau, dt):
        friction = self.model.frictionloss * torch.tanh(qd / 0.05)  # smooth dry friction (zero in the nominal model)
        return self.model.step_contact(q, qd, tau - friction, dt, self.contact)


def rollout(policy: Policy, dynamics: DynamicsInterface, q_ref, qd_ref, q0, qd0, cfg: RolloutConfig,
            loss_cfg: LossConfig | None = None, train: bool = False, latency_s: float | None = None,
            obs_noise_q: float = 0.0, f_ext=None) -> dict:
    """Roll the closed loop over q_ref/qd_ref (B,T,7). If ``train``, backpropagates the loss chunk by
    chunk. Returns per-step logs (detached) and loss terms averaged over the batch and horizon.
    ``latency_s`` overrides the policy's nominal latency and ``obs_noise_q`` adds measurement noise
    (position; velocity noise is 10x) to what the network and the fast loop see (randomization).
    ``f_ext`` (B,T,3), if given, is an external world-frame force at the tool (disturbance pulses),
    applied through the tool Jacobian; the controller never observes it directly."""
    B, T, _ = q_ref.shape
    dev, dtype = q_ref.device, q_ref.dtype
    limits = policy.limits.to(dtype)
    period = int(round(1.0 / (policy.cfg.rate_hz * cfg.dt)))
    latency = int(round((policy.cfg.latency_s if latency_s is None else latency_s) / cfg.dt))
    latency = min(max(latency, 0), period - 1)
    H = policy.cfg.history
    q, qd = q0.clone(), qd0.clone()
    hist_q = q0[:, None].repeat(1, H, 1)
    hist_qd = qd0[:, None].repeat(1, H, 1)
    hist_tau = torch.zeros(B, H, NUM_JOINTS, device=dev, dtype=dtype)
    last_tau = torch.zeros(B, NUM_JOINTS, device=dev, dtype=dtype)
    held = {k: v.to(dtype) for k, v in policy.neutral_output(B).items()}
    gravity = dynamics.gravity(q)
    pending = None
    logs = {k: [] for k in ("q", "qd", "tau", "f", "tool_pos")}
    terms = {k: 0.0 for k in ("pos", "vel", "torque", "rate", "force", "impulse")}
    chunk = cfg.tbptt_steps if cfg.tbptt_steps > 0 else T
    prev_f_mag = torch.zeros(B, device=dev, dtype=dtype)
    for k0 in range(0, T, chunk):
        k1 = min(T, k0 + chunk)
        chunk_terms = {k: torch.zeros((), device=dev, dtype=dtype) for k in terms}
        for k in range(k0, k1):
            if obs_noise_q > 0.0:
                q_obs = q + obs_noise_q * torch.randn_like(q)
                qd_obs = qd + 10.0 * obs_noise_q * torch.randn_like(qd)
            else:
                q_obs, qd_obs = q, qd
            if k % period == 0:
                hist_q = torch.cat([hist_q[:, 1:], q_obs[:, None]], 1)
                hist_qd = torch.cat([hist_qd[:, 1:], qd_obs[:, None]], 1)
                hist_tau = torch.cat([hist_tau[:, 1:], last_tau[:, None]], 1)
                obs = encode_obs(q_obs, qd_obs, q_ref[:, k], qd_ref[:, k], hist_q, hist_qd, hist_tau, limits)
                pending = (k + latency, policy.decode(policy(obs.to(torch.float32)).to(dtype)))
                gravity = dynamics.gravity(q_obs)
            if pending is not None and k >= pending[0]:
                held, pending = pending[1], None
            tau = policy.fast_torque(held, q_obs, qd_obs, q_ref[:, k], qd_ref[:, k], gravity)
            tau = torch.maximum(torch.minimum(tau, limits), -limits)
            e, ed = q_ref[:, k] - q, qd_ref[:, k] - qd
            tau_plant = tau
            if f_ext is not None:
                tau_plant = tau + torch.einsum("bij,bi->bj", dynamics.tool_jacobian(q), f_ext[:, k])
            q, qd, f, p = dynamics.step(q, qd, tau_plant, cfg.dt)
            f_mag = torch.linalg.norm(f, dim=-1)
            if loss_cfg is not None:
                c = loss_cfg
                chunk_terms["pos"] = chunk_terms["pos"] + (e / c.pos_scale).square().sum(-1).mean()
                chunk_terms["vel"] = chunk_terms["vel"] + c.vel_weight * (ed / c.vel_scale).square().sum(-1).mean()
                chunk_terms["torque"] = chunk_terms["torque"] + c.torque_weight * (tau / limits).square().sum(-1).mean()
                chunk_terms["rate"] = chunk_terms["rate"] + c.rate_weight * ((tau - last_tau) / (cfg.dt * c.rate_scale)).square().sum(-1).mean()
                chunk_terms["force"] = chunk_terms["force"] + c.force_weight * (torch.relu(f_mag - c.force_soft) / c.force_scale).square().mean()
                chunk_terms["impulse"] = chunk_terms["impulse"] + c.impulse_weight * ((f_mag - prev_f_mag) / (cfg.dt * c.impulse_scale)).square().mean()
            last_tau, prev_f_mag = tau, f_mag
            for name, val in (("q", q), ("qd", qd), ("tau", tau), ("f", f), ("tool_pos", p)):
                logs[name].append(val.detach())
        if loss_cfg is not None:
            chunk_loss = sum(chunk_terms.values()) / T
            if train:
                chunk_loss.backward()
            for name in terms:
                terms[name] += chunk_terms[name].item() / T
        # truncate the graph at the chunk boundary
        q, qd, last_tau, prev_f_mag = q.detach(), qd.detach(), last_tau.detach(), prev_f_mag.detach()
        hist_q, hist_qd, hist_tau = hist_q.detach(), hist_qd.detach(), hist_tau.detach()
        held = {k: v.detach() for k, v in held.items()}
        gravity = gravity.detach()
        if pending is not None:
            pending = (pending[0], {k: v.detach() for k, v in pending[1].items()})
    out = {name: torch.stack(vals, 1) for name, vals in logs.items()}
    out["loss_terms"] = terms
    out["loss"] = sum(terms.values())
    return out
