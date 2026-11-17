"""Differentiable batched MuJoCo stepping for torch (ADR 0005): the simulator is the training medium.

``MjBatch`` owns B ``PandaSim`` instances (one robot instance / payload per batch element) and exposes
one physics step (1 ms) as a ``torch.autograd.Function``.  The forward pass is ``mj_step``; the backward
pass uses the transition matrices from ``mujoco.mjd_transitionFD`` evaluated at the pre-step state,

    d x_next = A dx + B du,   d touch = C dx + D du,     x = [qpos, qvel] (14), u = ctrl (7),

so that for upstream gradients g_x on x_next and g_y on the touch sensor the input gradients are
``grad_x = A^T g_x + C^T g_y`` and ``grad_u = B^T g_x + D^T g_y`` (batched matmuls, float64).

Contract of ``step``: the simulators hold the true state.  The ``q``, ``qd`` tensors passed to ``step``
must be the ones returned by the previous ``step`` (or by ``state()`` after ``reset``); they are only
used to thread the autograd graph through the rollout (this is checked).  The caller never writes the
simulator state except through ``reset``.  Observation noise, references and policies live outside
this module; the tool position is returned as a plain numpy array because callers use the analytic
``RigidBodyModel`` for differentiable kinematics.

There is no separate functional wrapper: ``batch.step(q, qd, tau, f_ext)`` is the only entry point.

Threading: the MuJoCo bindings release the GIL during ``mj_step`` and ``mjd_transitionFD``, so the
batch is stepped by a persistent thread pool, each worker owning a contiguous chunk of simulators.
Measured on a 15-core Apple Silicon laptop with sampled robot instances (dry friction adds constraint
rows) and forward differences, ms per batch step (free space / in contact):
B = 24: 1.8 / 1.6 with one thread, 0.65 / 0.54 with four, 0.58 / 0.58 with eight, 0.77 / 0.64 with 14;
B = 48: 3.6 / 5.0, 1.7 / 1.7, 1.4 / 1.7, 1.4 / 2.0.  A 500-step forward + backward rollout with a small
MLP policy and torch gravity compensation takes about 1.3 s (B = 24) and 1.6 s (B = 48) with eight
threads.  More than about eight threads does not help because the per-simulator Python calls still
contend for the GIL; building the B simulators costs about 1 ms each.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import torch

from llc.sim.panda import TORQUE_LIMITS, DynamicsPerturbation, PandaSim

DEFAULT_THREADS = 8


class MjBatch:
    """B MuJoCo simulators stepped together, differentiable through finite differences.

    ``perturbations``: one ``DynamicsPerturbation`` per batch element (``actuation_delay`` must be zero:
    the torque is applied for the step it is given, as in ADR 0005).  ``eps`` and ``centered`` are passed
    to ``mjd_transitionFD`` (forward differences by default).  ``threads`` is the pool size (1 = no pool).
    """

    def __init__(self, perturbations: list[DynamicsPerturbation], seed: int = 0, eps: float = 1e-6,
                 centered: bool = False, threads: int | None = None):
        self.seed, self.eps, self.centered = seed, float(eps), bool(centered)
        self.sims: list[PandaSim] = []
        self.perturb(perturbations)
        self.B = len(self.sims)
        m = self.sims[0].model
        self.nv, self.nu = int(m.nv), int(m.nu)
        nx = 2 * self.nv
        self._touch_adr = int(m.sensor_adr[self.sims[0].touch_sensor])
        self._tool_site = self.sims[0].tool_site
        # Per-step buffers, written by the workers at their batch index.
        self._tau = np.zeros((self.B, self.nu))
        self._x = np.zeros((self.B, nx))
        self._touch = np.zeros(self.B)
        self._tool = np.zeros((self.B, 3))
        self._A = np.zeros((self.B, nx, nx))
        self._B = np.zeros((self.B, nx, self.nu))
        self._C = np.zeros((self.B, 1, nx))
        self._D = np.zeros((self.B, 1, self.nu))
        self._need_fd = True
        self._f_ext_active = False
        self._limits = torch.as_tensor(TORQUE_LIMITS, dtype=torch.float32)
        n = max(1, min(self.B, DEFAULT_THREADS if threads is None else int(threads)))
        self._pool = ThreadPoolExecutor(n) if n > 1 else None
        edges = np.linspace(0, self.B, n + 1).astype(int)
        self._chunks = [(int(lo), int(hi)) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]

    # ------------------------------------------------------------------ setup and state
    def perturb(self, perturbations: list[DynamicsPerturbation]) -> None:
        """Replace the dynamics perturbations (robot instances, payloads) of all simulators, keeping (q, qd).

        Rebuilds the simulators (about 1 ms each: the scene compiles quickly), so this is cheap enough to
        call between training windows.  The solver warm start and the simulation time are reset.
        """
        if self.sims and len(perturbations) != len(self.sims):
            raise ValueError("the batch size cannot change")
        if any(p.actuation_delay for p in perturbations):
            raise ValueError("MjBatch applies torques with zero latency; actuation_delay must be 0")
        q, qd = self._state_np() if self.sims else (None, None)
        self.sims = [PandaSim(p, seed=self.seed + i) for i, p in enumerate(perturbations)]
        if q is not None:
            self.reset(q, qd)

    def reset(self, q0: np.ndarray, qd0: np.ndarray | None = None) -> None:
        """Set the state of every simulator: ``q0`` (B, 7) and optionally ``qd0`` (B, 7)."""
        for i, s in enumerate(self.sims):
            s.reset(np.asarray(q0[i], dtype=float), None if qd0 is None else np.asarray(qd0[i], dtype=float))
        self._f_ext_active = False

    def _state_np(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array([s.data.qpos for s in self.sims]), np.array([s.data.qvel for s in self.sims])

    def state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """True joint state (q, qd) as float32 tensors (B, 7), copied from the simulators."""
        q, qd = self._state_np()
        return torch.as_tensor(q, dtype=torch.float32), torch.as_tensor(qd, dtype=torch.float32)

    # ------------------------------------------------------------------ stepping
    def step(self, q: torch.Tensor, qd: torch.Tensor, tau: torch.Tensor, f_ext: np.ndarray | None = None):
        """One physics step with torque ``tau`` (B, 7), clamped to the actuator limits (zero gradient
        beyond them), and an optional external world-frame force ``f_ext`` (B, 3) at the tool body for
        this step only.  Returns ``(q_next, qd_next, touch, tool_pos)``: the first three are float32
        tensors differentiable with respect to ``q``, ``qd`` and ``tau``; ``touch`` (B,) is the normal
        contact force on the tool tip (``tool_touch`` sensor); ``tool_pos`` (B, 3) is a numpy copy."""
        tau = torch.clamp(tau, -self._limits, self._limits)
        self._need_fd = torch.is_grad_enabled() and (q.requires_grad or qd.requires_grad or tau.requires_grad)
        q_next, qd_next, touch = _MjStep.apply(self, q, qd, tau, f_ext)
        return q_next, qd_next, touch, self._tool.copy()

    def _step_all(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, f_ext: np.ndarray | None) -> None:
        """Run the batch step (and finite differences if needed) into the buffers."""
        q_sim, qd_sim = self._state_np()
        if not (np.allclose(q, q_sim, atol=1e-5) and np.allclose(qd, qd_sim, atol=1e-5)):
            raise ValueError("step received a state that differs from the simulators' state (see module doc)")
        np.copyto(self._tau, tau)
        if f_ext is not None or self._f_ext_active:
            for i, s in enumerate(self.sims):
                s.apply_tool_force(np.zeros(3) if f_ext is None else np.asarray(f_ext[i], dtype=float))
            self._f_ext_active = f_ext is not None
        if self._pool is None:
            self._step_chunk(self._chunks[0])
        else:
            list(self._pool.map(self._step_chunk, self._chunks))

    def _step_chunk(self, bounds: tuple[int, int]) -> None:
        nv, adr, site, eps, centered, fd = self.nv, self._touch_adr, self._tool_site, self.eps, self.centered, self._need_fd
        for i in range(*bounds):
            s = self.sims[i]
            d = s.data
            d.ctrl[:] = self._tau[i]
            if fd:
                mujoco.mjd_transitionFD(s.model, d, eps, centered, self._A[i], self._B[i], self._C[i], self._D[i])
            mujoco.mj_step(s.model, d)
            self._x[i, :nv] = d.qpos
            self._x[i, nv:] = d.qvel
            self._touch[i] = d.sensordata[adr]
            self._tool[i] = d.site_xpos[site]


class _MjStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, batch: MjBatch, q, qd, tau, f_ext):
        batch._step_all(q.detach().numpy(), qd.detach().numpy(), tau.detach().to(torch.float64).numpy(), f_ext)
        if batch._need_fd:
            ctx.save_for_backward(*(torch.from_numpy(a.copy()) for a in (batch._A, batch._B, batch._C, batch._D)))
        ctx.dtypes = (q.dtype, qd.dtype, tau.dtype)
        ctx.touch_local = getattr(batch, "touch_grad_local", False)
        nv = batch.nv
        x = torch.from_numpy(batch._x.astype(np.float32))
        return x[:, :nv], x[:, nv:], torch.from_numpy(batch._touch.astype(np.float32))

    @staticmethod
    def backward(ctx, g_q, g_qd, g_touch):
        A, B, C, D = ctx.saved_tensors
        g_x = torch.cat([g_q, g_qd], -1).to(torch.float64).unsqueeze(-1)  # (B, 14, 1)
        g_y = g_touch.to(torch.float64).view(-1, 1, 1)  # (B, 1, 1)
        if ctx.touch_local:  # contact-force gradient only through the current control input (ADR 0005 amendment)
            grad_x = (A.transpose(1, 2) @ g_x).squeeze(-1)
        else:
            grad_x = (A.transpose(1, 2) @ g_x + C.transpose(1, 2) @ g_y).squeeze(-1)
        grad_u = (B.transpose(1, 2) @ g_x + D.transpose(1, 2) @ g_y).squeeze(-1)
        dq, dqd, dtau = ctx.dtypes
        nv = grad_x.shape[1] // 2
        return None, grad_x[:, :nv].to(dq), grad_x[:, nv:].to(dqd), grad_u.to(dtau), None
