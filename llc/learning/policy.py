"""Learned low-level controller parameterizations (PyTorch).

A policy network runs at the network rate (default 100 Hz, ADR 0002) on an observation made of
the current measured state, the current reference and a history of past (q, qd, applied torque)
samples, and outputs one of

- learned_gains:  per-joint stiffness and damping for a PD + gravity law,
- pd_residual:    a bounded residual torque added to the shared-tuned PD + gravity law,
- neural_torque:  the joint torque itself.

The network output is held between updates and applied with a compute latency. The fast loop
(``fast_torque``) turns the held output into a torque at every physics step. The same encoding
and decoding functions are used by the simulator controller and by the differentiable training
rollout, so the two never drift apart.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn

from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS

KINDS = ("learned_gains", "pd_residual", "neural_torque", "pd_gain_residual")


@dataclass
class PolicyConfig:
    kind: str
    history: int = 10  # samples at the network rate
    rate_hz: float = 100.0
    latency_s: float = 0.005
    hidden: int = 128
    layers: int = 2
    omega_range: tuple[float, float] = (1.0, 40.0)  # rad/s, natural frequency of the learned PD per joint
    omega_init: float = 15.0
    zeta_range: tuple[float, float] = (0.2, 3.0)
    zeta_init: float = 1.0
    residual_fraction: float = 0.5  # residual authority as a fraction of the torque limit
    inertia_diag: tuple[float, ...] = field(default_factory=lambda: (1.0,) * NUM_JOINTS)  # nominal M_ii at home, for the gain map
    kp_shared: tuple[float, ...] = field(default_factory=lambda: (0.0,) * NUM_JOINTS)  # pd_residual only
    kd_shared: tuple[float, ...] = field(default_factory=lambda: (0.0,) * NUM_JOINTS)
    # Deep-dive history (ADR 0005): ``history`` samples spaced ``history_stride`` network steps apart
    # plus ``history_dense`` consecutive samples. With rate_hz = 1000, history=10, stride=10 and
    # history_dense=32 this is 100 ms at 10 ms spacing plus the last 32 ms at 1 ms.
    history_stride: int = 1
    history_dense: int = 0
    encoding: str = "raw"  # "raw" or "fourier" (random Fourier features on error, velocity and history)
    fourier_features: int = 64
    # Position-target history (same two-scale windows), encoded relative to the current target. Gives
    # the controller the target's own motion, hence its acceleration and the comparison between
    # where the target went and where the robot went (a contact signal). Position targets only.
    ref_history: bool = False
    # Decision period (deep dive D4): the network output is held for ``hold_steps`` physics steps while
    # the history is still sampled at every physics step; 1 reproduces the 1 kHz policies.
    hold_steps: int = 1
    # Contact deep dive (ADR 0007). ``pd_gain_residual``: the network scales the shared PD gains per
    # joint (bounded multipliers, neutral at 1) and adds a torque residual; yielding and holding are
    # then small actions and the closed loop stays a positive-stiffness, positive-damping PD.
    gain_scale_range: tuple[float, float] = (0.02, 1.5)
    damping_scale_range: tuple[float, float] = (0.3, 2.0)
    # External-torque estimate from the robot's own rigid-body model, tau_applied - (M qdd + bias), in
    # the history windows: the signal a torque-controlled arm provides on hardware (Panda: tau_ext_hat).
    ext_torque: bool = False
    # First-order low-pass of the held output at the physics rate (seconds; 0 = none): smooth torques.
    output_tau: float = 0.0
    # Target preview (ADR 0008, variant): the next ``preview`` position targets, ``preview_stride``
    # physics steps apart, encoded relative to the current target. Gives the controller the target's
    # upcoming motion (anticipatory feedforward) and a cleaner blocked cue (the target keeps moving
    # while the robot does not). 0 = none.
    preview: int = 0
    preview_stride: int = 10
    # Feedforward base (ADR 0010 round 2): the PD base of pd_residual / pd_gain_residual includes the nominal
    # inverse-dynamics feedforward of the reference, M(q) qdd_ref + C(q, qd_ref) qd_ref + g(q), instead of
    # gravity alone; the fast loop's base term is supplied by the caller (nominal model or torch model).
    feedforward: bool = False

    @property
    def n_hist(self) -> int:
        return self.history + self.history_dense

    @property
    def raw_dim(self) -> int:
        return 4 * NUM_JOINTS + self.n_hist * (3 + int(self.ref_history) + int(self.ext_torque)) * NUM_JOINTS + self.preview * NUM_JOINTS

    @property
    def obs_dim(self) -> int:
        return self.raw_dim + 2 * self.fourier_features if self.encoding == "fourier" else self.raw_dim

    @property
    def out_dim(self) -> int:
        return {"learned_gains": 2, "pd_gain_residual": 3}.get(self.kind, 1) * NUM_JOINTS


# Fixed observation scales (not fitted to data): rad, rad/s and Nm into O(1) numbers.
Q_SCALE, QD_SCALE, ERR_SCALE = 3.0, 2.0, 0.2


def encode_obs(q, qd, q_ref, qd_ref, hist_q, hist_qd, hist_tau, limits, hist_ref=None, hist_ext=None, ref_future=None) -> torch.Tensor:
    """Observation vector. Shapes: (B,7) for the current quantities, (B,H,7) for the history, (B,P,7)
    for the target preview. History positions are encoded relative to the current position; the
    optional target history and the preview relative to the current target."""
    cur = torch.cat([q / Q_SCALE, qd / QD_SCALE, (q_ref - q) / ERR_SCALE, qd_ref / QD_SCALE], dim=-1)
    parts = [(hist_q - q[:, None, :]) / ERR_SCALE, hist_qd / QD_SCALE, hist_tau / limits]
    if hist_ref is not None:
        parts.append((hist_ref - q_ref[:, None, :]) / ERR_SCALE)
    if hist_ext is not None:
        parts.append(hist_ext / limits)
    obs = [cur, torch.cat(parts, dim=-1).flatten(1)]
    if ref_future is not None:
        obs.append(((ref_future - q_ref[:, None, :]) / ERR_SCALE).flatten(1))
    return torch.cat(obs, dim=-1)


def preview_indices(cfg: PolicyConfig, k: int, T: int) -> np.ndarray:
    """Reference indices of the target preview at step ``k`` of a reference of ``T`` steps (held at the end)."""
    return np.minimum(k + cfg.preview_stride * np.arange(1, cfg.preview + 1), T - 1)


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


class Policy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        assert cfg.kind in KINDS, cfg.kind
        self.cfg = cfg
        if cfg.encoding == "fourier":
            g = torch.Generator().manual_seed(0)
            self.register_buffer("fourier_B", torch.randn(cfg.raw_dim, cfg.fourier_features, generator=g) * 2.0)
        dims = [cfg.obs_dim] + [cfg.hidden] * cfg.layers
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ELU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.hidden, cfg.out_dim)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)  # start neutral: initial gains / zero residual / zero torque
        self.register_buffer("limits", torch.tensor(TORQUE_LIMITS, dtype=torch.float32))
        self.register_buffer("inertia_diag", torch.tensor(cfg.inertia_diag, dtype=torch.float32))
        self.register_buffer("kp_shared", torch.tensor(cfg.kp_shared, dtype=torch.float32))
        self.register_buffer("kd_shared", torch.tensor(cfg.kd_shared, dtype=torch.float32))
        lo, hi = np.log(cfg.omega_range[0]), np.log(cfg.omega_range[1])
        self._omega_bias = _logit((np.log(cfg.omega_init) - lo) / (hi - lo))
        lo, hi = np.log(cfg.zeta_range[0]), np.log(cfg.zeta_range[1])
        self._zeta_bias = _logit((np.log(cfg.zeta_init) - lo) / (hi - lo))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.cfg.encoding == "fourier":
            proj = obs @ self.fourier_B
            obs = torch.cat([obs, torch.sin(proj), torch.cos(proj)], -1)
        return self.head(self.trunk(obs))

    def decode(self, out: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map raw network outputs to physical quantities (gains in Nm/rad and Nm s/rad, or torques in Nm)."""
        cfg = self.cfg
        if cfg.kind == "learned_gains":
            lo, hi = np.log(cfg.omega_range)
            omega = torch.exp(lo + (hi - lo) * torch.sigmoid(out[..., :NUM_JOINTS] + self._omega_bias))
            lo, hi = np.log(cfg.zeta_range)
            zeta = torch.exp(lo + (hi - lo) * torch.sigmoid(out[..., NUM_JOINTS:] + self._zeta_bias))
            kp = self.inertia_diag * omega**2
            kd = 2.0 * zeta * omega * self.inertia_diag
            return {"kp": kp, "kd": kd}
        if cfg.kind == "pd_residual":
            return {"residual": cfg.residual_fraction * self.limits * torch.tanh(out)}
        if cfg.kind == "pd_gain_residual":
            n = NUM_JOINTS
            lo, hi = np.log(cfg.gain_scale_range)
            kp_scale = torch.exp(lo + (hi - lo) * torch.sigmoid(out[..., :n] + _logit(-lo / (hi - lo))))
            lo, hi = np.log(cfg.damping_scale_range)
            kd_scale = torch.exp(lo + (hi - lo) * torch.sigmoid(out[..., n:2 * n] + _logit(-lo / (hi - lo))))
            return {"kp_scale": kp_scale, "kd_scale": kd_scale, "residual": cfg.residual_fraction * self.limits * torch.tanh(out[..., 2 * n:])}
        return {"torque": self.limits * torch.tanh(out)}

    def fast_torque(self, held: dict[str, torch.Tensor], q, qd, q_ref, qd_ref, gravity) -> torch.Tensor:
        """Torque at a physics step from the held network output (torch, batched)."""
        kind = self.cfg.kind
        if kind == "learned_gains":
            return held["kp"] * (q_ref - q) + held["kd"] * (qd_ref - qd) + gravity
        if kind == "pd_residual":
            return self.kp_shared * (q_ref - q) + self.kd_shared * (qd_ref - qd) + gravity + held["residual"]
        if kind == "pd_gain_residual":
            return held["kp_scale"] * self.kp_shared * (q_ref - q) + held["kd_scale"] * self.kd_shared * (qd_ref - qd) + gravity + held["residual"]
        return held["torque"]

    def neutral_output(self, batch: int) -> dict[str, torch.Tensor]:
        """Held output before the first network update has taken effect."""
        return self.decode(torch.zeros(batch, self.cfg.out_dim, device=self.limits.device))


def save_policy(path, policy: Policy, extra: dict | None = None) -> None:
    torch.save({"config": asdict(policy.cfg), "state_dict": policy.state_dict(), "extra": extra or {}}, path)


def load_policy(path) -> tuple[Policy, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PolicyConfig(**ckpt["config"])
    policy = Policy(cfg)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy, ckpt.get("extra", {})


class HistoryBuffer:
    """Ring buffer of past (q, qd, tau) samples at the network rate, returning the two-scale history
    of ``PolicyConfig`` (strided samples oldest first, then dense samples oldest first). Works for
    torch tensors (B, 7) and is shared by the simulator controller and the training rollout."""

    def __init__(self, cfg: PolicyConfig, q0: torch.Tensor, qd0: torch.Tensor, inverse_dynamics=None, dt: float = 1e-3):
        self.cfg = cfg
        self.length = max(cfg.history_stride * (cfg.history - 1) + 1, cfg.history_dense, 1)
        B = q0.shape[0]
        self.q = q0[:, None].repeat(1, self.length, 1).clone()
        self.qd = qd0[:, None].repeat(1, self.length, 1).clone()
        self.tau = torch.zeros(B, self.length, NUM_JOINTS, dtype=q0.dtype, device=q0.device)
        self.ref: torch.Tensor | None = None  # target history, filled with the first pushed target
        # external-torque estimate tau_applied - (M(q) qdd + bias(q, qd)) from the model's inverse dynamics (B,7)x3 -> (B,7)
        self.ext = torch.zeros(B, self.length, NUM_JOINTS, dtype=q0.dtype, device=q0.device) if cfg.ext_torque else None
        self.inverse_dynamics, self.dt, self._pushes = inverse_dynamics, dt, 0
        assert not cfg.ext_torque or inverse_dynamics is not None, "ext_torque policies need the model's inverse dynamics"
        self._strided_idx = torch.arange(self.length - 1 - cfg.history_stride * (cfg.history - 1), self.length, cfg.history_stride) if cfg.history > 0 else torch.zeros(0, dtype=torch.long)
        self._dense_idx = torch.arange(self.length - cfg.history_dense, self.length) if cfg.history_dense > 0 else torch.zeros(0, dtype=torch.long)
        self._idx = torch.cat([self._strided_idx, self._dense_idx])

    def push(self, q: torch.Tensor, qd: torch.Tensor, tau: torch.Tensor, q_ref: torch.Tensor | None = None) -> None:
        if self.ext is not None:
            qdd = (qd - self.qd[:, -1]) / self.dt
            with torch.no_grad():  # an input feature, not a gradient path
                ext = tau - self.inverse_dynamics(q, qd, qdd) if self._pushes > 0 else torch.zeros_like(q)
            self.ext = torch.cat([self.ext[:, 1:], ext[:, None]], 1)
        self._pushes += 1
        self.q = torch.cat([self.q[:, 1:], q[:, None]], 1)
        self.qd = torch.cat([self.qd[:, 1:], qd[:, None]], 1)
        self.tau = torch.cat([self.tau[:, 1:], tau[:, None]], 1)
        if self.cfg.ref_history:
            assert q_ref is not None, "ref_history policies need the target at every push"
            self.ref = q_ref[:, None].repeat(1, self.length, 1).clone() if self.ref is None else torch.cat([self.ref[:, 1:], q_ref[:, None]], 1)

    def detach(self) -> None:
        self.q, self.qd, self.tau = self.q.detach(), self.qd.detach(), self.tau.detach()
        if self.ref is not None:
            self.ref = self.ref.detach()
        if self.ext is not None:
            self.ext = self.ext.detach()

    def windows(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q[:, self._idx], self.qd[:, self._idx], self.tau[:, self._idx]

    def observation(self, q, qd, q_ref, qd_ref, limits, ref_future=None) -> torch.Tensor:
        """Policy observation from the current quantities and the buffered windows (after push);
        ``ref_future`` (B, preview, 7) is required by preview policies."""
        hq, hqd, htau = self.windows()
        href = self.ref[:, self._idx] if self.cfg.ref_history else None
        hext = self.ext[:, self._idx] if self.ext is not None else None
        assert (ref_future is not None) == (self.cfg.preview > 0), "preview policies need the future targets at every observation"
        return encode_obs(q, qd, q_ref, qd_ref, hq, hqd, htau, limits, href, hext, ref_future)


class OutputFilter:
    """First-order low-pass of the held network output applied at every physics step
    (``PolicyConfig.output_tau`` seconds; 0 passes the output through). Works for torch tensors
    and numpy arrays; the state starts at the first output (no start-up transient)."""

    def __init__(self, tau_s: float, dt: float = 1e-3):
        self.alpha = min(1.0, dt / tau_s) if tau_s > 0 else 1.0
        self.state: dict | None = None

    def __call__(self, held: dict) -> dict:
        if self.alpha >= 1.0:
            return held
        if self.state is None:
            self.state = {k: v * 1.0 for k, v in held.items()}
        else:
            self.state = {k: self.state[k] + self.alpha * (v - self.state[k]) for k, v in held.items()}
        return self.state

    def detach(self) -> None:
        if self.state is not None:
            self.state = {k: (v.detach() if hasattr(v, "detach") else v) for k, v in self.state.items()}


def expand_ext_torque(policy: "Policy") -> "Policy":
    """The same policy with the external-torque feature added to its observation: the first layer's
    new input columns start at zero, so the warm start is reproduced exactly until training uses them."""
    cfg = policy.cfg
    assert not cfg.ext_torque and cfg.encoding == "raw", "expansion is defined for raw-encoded policies without the feature"
    new = Policy(replace(cfg, ext_torque=True))
    state = {k: v for k, v in policy.state_dict().items() if k != "trunk.0.weight"}
    new.load_state_dict(state, strict=False)
    per_old = (3 + int(cfg.ref_history)) * NUM_JOINTS
    cols = list(range(4 * NUM_JOINTS)) + [4 * NUM_JOINTS + i * (per_old + NUM_JOINTS) + j for i in range(cfg.n_hist) for j in range(per_old)]
    with torch.no_grad():
        new.trunk[0].weight.zero_()
        new.trunk[0].weight[:, cols] = policy.trunk[0].weight
    return new.eval()


def expand_preview(policy: "Policy", preview: int, preview_stride: int = 10) -> "Policy":
    """The same policy with a target preview added to its observation (new last input columns at
    zero, so the warm start is reproduced exactly until training uses them)."""
    cfg = policy.cfg
    assert cfg.preview == 0 and cfg.encoding == "raw", "expansion is defined for raw-encoded policies without a preview"
    new = Policy(replace(cfg, preview=preview, preview_stride=preview_stride))
    state = {k: v for k, v in policy.state_dict().items() if k != "trunk.0.weight"}
    new.load_state_dict(state, strict=False)
    with torch.no_grad():
        new.trunk[0].weight.zero_()
        new.trunk[0].weight[:, : cfg.obs_dim] = policy.trunk[0].weight
    return new.eval()


def convert_to_gain_residual(policy: "Policy", residual_fraction: float | None = None) -> "Policy":
    """A ``pd_gain_residual`` policy that reproduces a ``pd_residual`` policy exactly: same trunk,
    the residual head copied, the gain heads at their neutral output (multipliers 1)."""
    assert policy.cfg.kind == "pd_residual"
    cfg = replace(policy.cfg, kind="pd_gain_residual", residual_fraction=policy.cfg.residual_fraction if residual_fraction is None else residual_fraction)
    new = Policy(cfg)
    new.load_state_dict({k: v for k, v in policy.state_dict().items() if not k.startswith("head.")}, strict=False)
    with torch.no_grad():
        new.head.weight.zero_(); new.head.bias.zero_()
        scale = policy.cfg.residual_fraction / cfg.residual_fraction  # tanh outputs cannot be rescaled exactly; equal fractions are exact
        assert abs(scale - 1.0) < 1e-9, "exact conversion needs the same residual fraction"
        new.head.weight[2 * NUM_JOINTS:] = policy.head.weight
        new.head.bias[2 * NUM_JOINTS:] = policy.head.bias
    return new.eval()
