"""Closed-loop training rollout through the MuJoCo batch (ADR 0005): the policy runs at every
physics step with zero latency, the history is pushed every step, and gradients flow through
the finite-difference autograd step of ``llc.learning.mjgrad``. Loss terms as in
``llc.learning.rollout.LossConfig``; the contact force is the touch sensor."""
from __future__ import annotations

import numpy as np
import torch

from llc.learning.policy import HistoryBuffer, Policy
from llc.learning.rollout import LossConfig
from llc.sim.panda import NUM_JOINTS


def _clip_hook(max_norm: float):
    """Per-sample clipping of the gradient flowing backward through a state tensor (B, n): keeps the
    chained finite-difference Jacobians through contact from blowing up (ADR 0005)."""

    def hook(g):
        if g is None:
            return None
        n = g.norm(dim=-1, keepdim=True)
        return g * torch.clamp(max_norm / (n + 1e-12), max=1.0)

    return hook


def rollout_mj(policy: Policy, batch, gravity_model, q_ref, qd_ref, q0, qd0, f_ext=None, loss_cfg: LossConfig | None = None,
               train: bool = False, tbptt_steps: int = 100, obs_noise_q: float = 1e-4, dt: float = 1e-3, state_grad_clip: float = 1e3) -> dict:
    """q_ref/qd_ref (B,T,7) torch; q0/qd0 (B,7) numpy; f_ext (B,T,3) numpy or None. ``batch`` is an
    MjBatch already carrying the per-window perturbations; ``gravity_model`` provides the nominal
    gravity torque bias(q, 0) differentiably (RigidBodyModel)."""
    B, T, _ = q_ref.shape
    limits = policy.limits
    batch.reset(np.asarray(q0, dtype=np.float64), np.asarray(qd0, dtype=np.float64))
    q, qd = batch.state()
    q, qd = q.float(), qd.float()
    buffer = HistoryBuffer(policy.cfg, q, qd)
    last_tau = torch.zeros(B, NUM_JOINTS)
    prev_touch = torch.zeros(B)
    logs = {k: [] for k in ("q", "qd", "tau", "touch", "tool_pos")}
    terms = {k: 0.0 for k in ("pos", "vel", "torque", "rate", "force", "impulse")}
    chunk = tbptt_steps if tbptt_steps > 0 else T
    for k0 in range(0, T, chunk):
        k1 = min(T, k0 + chunk)
        ct = {k: torch.zeros(()) for k in terms}
        for k in range(k0, k1):
            if obs_noise_q > 0.0:
                q_obs = q + obs_noise_q * torch.randn_like(q)
                qd_obs = qd + 10.0 * obs_noise_q * torch.randn_like(qd)
            else:
                q_obs, qd_obs = q, qd
            buffer.push(q_obs, qd_obs, last_tau, q_ref[:, k])
            obs = buffer.observation(q_obs, qd_obs, q_ref[:, k], qd_ref[:, k], limits)
            held = policy.decode(policy(obs))
            gravity = gravity_model.bias(q_obs, torch.zeros_like(q_obs))
            if getattr(policy.cfg, "feedforward", False):
                from llc.learning.actor_critic import reference_accel

                gravity = gravity_model.inverse_dynamics(q_obs, qd_ref[:, k], reference_accel(qd_ref, 1e-3)[:, k])
            tau = policy.fast_torque(held, q_obs, qd_obs, q_ref[:, k], qd_ref[:, k], gravity)
            tau = torch.maximum(torch.minimum(tau, limits), -limits)
            e, ed = q_ref[:, k] - q, qd_ref[:, k] - qd
            q, qd, touch, tool_pos = batch.step(q, qd, tau, None if f_ext is None else f_ext[:, k])
            q, qd, touch = q.float(), qd.float(), touch.float()
            if train and state_grad_clip > 0 and q.requires_grad:
                q.register_hook(_clip_hook(state_grad_clip))
                qd.register_hook(_clip_hook(state_grad_clip))
            if loss_cfg is not None:
                c = loss_cfg
                ct["pos"] = ct["pos"] + (e / c.pos_scale).square().sum(-1).mean()
                ct["vel"] = ct["vel"] + c.vel_weight * (ed / c.vel_scale).square().sum(-1).mean()
                ct["torque"] = ct["torque"] + c.torque_weight * (tau / limits).square().sum(-1).mean()
                ct["rate"] = ct["rate"] + c.rate_weight * ((tau - last_tau) / (dt * c.rate_scale)).square().sum(-1).mean()
                ct["force"] = ct["force"] + c.force_weight * (torch.relu(touch - c.force_soft) / c.force_scale).square().mean()
                ct["impulse"] = ct["impulse"] + c.impulse_weight * ((touch - prev_touch) / (dt * c.impulse_scale)).square().mean()
            last_tau, prev_touch = tau, touch
            logs["q"].append(q.detach()); logs["qd"].append(qd.detach()); logs["tau"].append(tau.detach()); logs["touch"].append(touch.detach()); logs["tool_pos"].append(torch.as_tensor(tool_pos))
        if loss_cfg is not None:
            chunk_loss = sum(ct.values()) / T
            if train:
                chunk_loss.backward()
            for name in terms:
                terms[name] += ct[name].item() / T
        q, qd, last_tau, prev_touch = q.detach(), qd.detach(), last_tau.detach(), prev_touch.detach()
        buffer.detach()
    out = {name: torch.stack(vals, 1) for name, vals in logs.items()}
    out["loss_terms"] = terms
    out["loss"] = sum(terms.values())
    return out
