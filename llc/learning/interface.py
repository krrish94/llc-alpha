"""Learned torque-command interface (ADR 0011, decision 4): one network per robot and profile that takes
the upper command frame and the history of the last host frames and emits bounded corrections on the
channels the profile has (a goal offset; gain multipliers where the gains are writable; a feedforward
torque residual where the channel exists), applied on top of the model-based interface, which it
reproduces exactly at its warm start (zero head).

The module has two halves that share one law: ``LearnedInterface`` (numpy, the simulator side; the
same call signature as ``ModelInterface``) and the torch functions used by the trainer
(``model_interface_torch``, ``servo_law_torch``, ``InterfacePolicy``). Quantisation is straight-through
in torch. ``rollout_interface`` runs the servo loop through the MuJoCo batch with the loss between the
realised torque and the upper command at the true state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from llc.learning.policy import ERR_SCALE, QD_SCALE
from llc.sim.panda import NUM_JOINTS, TORQUE_LIMITS
from llc.sim.servo import ModelInterface, ServoProfile, get_profile, quantise

LIMITS_T = torch.tensor(TORQUE_LIMITS, dtype=torch.float32)


@dataclass
class InterfaceConfig:
    profile: str
    kp_nominal: tuple[float, ...]
    kd_nominal: tuple[float, ...]
    history: int = 10  # host frames
    hidden: int = 256
    layers: int = 4
    goal_bound: float = 0.05  # rad, bound of the goal offset
    gain_range: tuple[float, float] = (0.5, 2.0)  # multipliers on the model-based servo gains (writable gains only)
    tau_bound: float = 0.2  # fraction of the torque limit, bound of the feedforward residual (torque channel only)
    overrides: dict | None = None  # profile overrides (period, delay)

    @property
    def obs_dim(self) -> int:
        return 6 * NUM_JOINTS + self.history * 4 * NUM_JOINTS

    @property
    def out_dim(self) -> int:
        return 4 * NUM_JOINTS


def profile_of(cfg: InterfaceConfig) -> ServoProfile:
    return get_profile(cfg.profile, **(cfg.overrides or {}))


# ---------------------------------------------------------------- torch forms of the servo layer

def st_quantise(x: torch.Tensor, res: float) -> torch.Tensor:
    """Straight-through quantisation: forward rounds, backward passes the gradient."""
    if res <= 0.0:
        return x
    return x + (torch.round(x / res) * res - x).detach()


def st_quantise_range(x: torch.Tensor, lo, hi, bits: int) -> torch.Tensor:
    x = torch.maximum(torch.minimum(x, hi), lo)
    if bits <= 0:
        return x
    n = 2**bits - 1
    q = lo + torch.round((x - lo) / (hi - lo + 1e-12) * n) / n * (hi - lo)
    return x + (q - x).detach()


def frame_law_torch(frame: dict, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    tau = frame["kp"] * (frame["q_des"] - q) + frame["kd"] * (frame["qd_des"] - qd) + frame["tau_ff"]
    return torch.maximum(torch.minimum(tau, frame["cap"]), -frame["cap"])


def servo_gains_torch(p: ServoProfile, kp_n: torch.Tensor, kd_n: torch.Tensor, frame: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if p.gains == "fixed":
        return kp_n.expand_as(frame["kp"]), kd_n.expand_as(frame["kd"])
    kp = torch.where(frame["kp"] > 0.0, frame["kp"], kp_n)
    kd = torch.where(frame["kp"] > 0.0, frame["kd"], kd_n)
    kp = st_quantise_range(kp, p.kp_range[0] * kp_n, p.kp_range[1] * kp_n, p.gain_bits)
    kd = st_quantise_range(kd, p.kd_range[0] * kd_n, p.kd_range[1] * kd_n, p.gain_bits)
    return kp, kd


def model_interface_torch(p: ServoProfile, kp_n: torch.Tensor, kd_n: torch.Tensor, frame: dict, q_m: torch.Tensor, qd_m: torch.Tensor) -> dict:
    """The model-based interface of ``llc.sim.servo.ModelInterface`` on batched torch frames (B,7)."""
    kp_s, kd_s = servo_gains_torch(p, kp_n, kd_n, frame)
    cap = torch.minimum(frame["cap"], LIMITS_T) if p.cap_cmd else LIMITS_T.expand_as(frame["cap"])
    frame = {**frame, "q_des": frame["q_des"] + frame["qd_des"] * (p.delay + 0.5 * (p.period - 1e-3))}  # the reference advanced to the middle of the hold (as the numpy form)
    if p.tau_ff and p.vel_cmd and p.gains == "frame":
        tau_ff = frame["tau_ff"] + (frame["kp"] - kp_s) * (frame["q_des"] - q_m) + (frame["kd"] - kd_s) * (frame["qd_des"] - qd_m)
        return {"q_des": st_quantise(frame["q_des"], p.goal_res), "qd_des": st_quantise(frame["qd_des"], p.vel_cmd_res), "kp": kp_s, "kd": kd_s,
                "tau_ff": st_quantise_range(tau_ff, -LIMITS_T, LIMITS_T, p.tau_bits), "cap": cap}
    q_p = q_m + qd_m * p.delay
    tau_c = frame_law_torch(frame, q_p, qd_m)
    q_goal = st_quantise(q_p + (tau_c + kd_s * qd_m) / kp_s, p.goal_res)
    z = torch.zeros_like(q_m)
    return {"q_des": q_goal, "qd_des": z, "kp": kp_s, "kd": kd_s, "tau_ff": z, "cap": cap}


# ---------------------------------------------------------------- the network

def interface_obs(cfg: InterfaceConfig, frame: dict, q_m, qd_m, hist_q, hist_qd, hist_tau, hist_goal, kp_n, kd_n):
    """Observation (B, obs_dim): the frame relative to the measured state, and the history of the last
    ``history`` host frames (positions relative to the current one, velocities, the realised torque and
    the goal sent, relative to the current position). Works for torch tensors."""
    cur = torch.cat([(frame["q_des"] - q_m) / ERR_SCALE, frame["qd_des"] / QD_SCALE, frame["kp"] / kp_n, frame["kd"] / kd_n, frame["tau_ff"] / LIMITS_T, frame["cap"] / LIMITS_T], -1)
    hist = torch.cat([(hist_q - q_m[:, None]) / ERR_SCALE, hist_qd / QD_SCALE, hist_tau / LIMITS_T, (hist_goal - q_m[:, None]) / ERR_SCALE], -1).flatten(1)
    return torch.cat([cur, hist], -1)


class InterfacePolicy(nn.Module):
    def __init__(self, cfg: InterfaceConfig):
        super().__init__()
        self.cfg = cfg
        dims = [cfg.obs_dim] + [cfg.hidden] * cfg.layers
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ELU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.hidden, cfg.out_dim)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)  # exact warm start: the model-based interface
        self.register_buffer("kp_n", torch.tensor(cfg.kp_nominal, dtype=torch.float32))
        self.register_buffer("kd_n", torch.tensor(cfg.kd_nominal, dtype=torch.float32))
        lo, hi = np.log(cfg.gain_range)
        self._gain_bias = float(np.log((0.0 - lo) / (hi - lo) / (1.0 - (0.0 - lo) / (hi - lo))))  # sigmoid(bias) maps to a multiplier of 1

    def forward(self, obs):
        return self.head(self.trunk(obs))

    def decode(self, out: torch.Tensor) -> dict[str, torch.Tensor]:
        n, cfg = NUM_JOINTS, self.cfg
        lo, hi = np.log(cfg.gain_range)
        return {"dq": cfg.goal_bound * torch.tanh(out[..., :n]),
                "m_kp": torch.exp(lo + (hi - lo) * torch.sigmoid(out[..., n:2 * n] + self._gain_bias)),
                "m_kd": torch.exp(lo + (hi - lo) * torch.sigmoid(out[..., 2 * n:3 * n] + self._gain_bias)),
                "dtau": cfg.tau_bound * LIMITS_T * torch.tanh(out[..., 3 * n:])}

    def apply(self, p: ServoProfile, servo: dict, corr: dict) -> dict:
        """The corrected servo frame (torch): goal offset; gain multipliers and torque residual where the channel exists."""
        out = dict(servo)
        out["q_des"] = st_quantise(servo["q_des"] + corr["dq"], p.goal_res)
        if p.gains != "fixed":
            out["kp"] = st_quantise_range(servo["kp"] * corr["m_kp"], p.kp_range[0] * self.kp_n, p.kp_range[1] * self.kp_n, p.gain_bits)
            out["kd"] = st_quantise_range(servo["kd"] * corr["m_kd"], p.kd_range[0] * self.kd_n, p.kd_range[1] * self.kd_n, p.gain_bits)
        if p.tau_ff:
            out["tau_ff"] = st_quantise_range(servo["tau_ff"] + corr["dtau"], -LIMITS_T, LIMITS_T, p.tau_bits)
        return out


def save_interface(path, policy: InterfacePolicy, extra: dict | None = None) -> None:
    torch.save({"config": asdict(policy.cfg), "state_dict": policy.state_dict(), "extra": extra or {}}, path)


def load_interface(path) -> tuple[InterfacePolicy, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = InterfaceConfig(**ckpt["config"])
    pol = InterfacePolicy(cfg)
    pol.load_state_dict(ckpt["state_dict"])
    return pol.eval(), ckpt.get("extra", {})


class LearnedInterface(ModelInterface):
    """The learned interface on the simulator side: the model-based mapping plus the network's corrections,
    with a history of the last host frames (measured state, realised torque, goal sent)."""

    def __init__(self, policy: InterfacePolicy, profile: ServoProfile, kp_nominal, kd_nominal):
        super().__init__(profile, kp_nominal, kd_nominal)
        self.policy = policy.eval()
        self.cfg = policy.cfg
        self.last_tau = np.zeros(NUM_JOINTS)  # set by the servo loop at every physics step
        self.reset()

    def reset(self) -> None:
        self.hist: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None
        self.last_goal = None

    def __call__(self, frame: dict, q_m: np.ndarray, qd_m: np.ndarray) -> dict:
        servo = super().__call__(frame, q_m, qd_m)
        H = self.cfg.history
        if self.hist is None:
            self.hist = [(q_m.copy(), qd_m.copy(), np.zeros(NUM_JOINTS), q_m.copy())] * H
        self.hist = self.hist[1:] + [(q_m.copy(), qd_m.copy(), self.last_tau.copy(), (self.last_goal if self.last_goal is not None else q_m).copy())]
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)[None]
        ft = {k: f(v) for k, v in frame.items()}
        hq, hqd, htau, hgoal = (torch.stack([f(h[i])[0] for h in self.hist])[None] for i in range(4))
        with torch.no_grad():
            obs = interface_obs(self.cfg, ft, f(q_m), f(qd_m), hq, hqd, htau, hgoal, self.policy.kp_n, self.policy.kd_n)
            corr = self.policy.decode(self.policy(obs))
            out = self.policy.apply(self.p, {k: f(v) for k, v in servo.items()}, corr)
        out = {k: v[0].numpy().astype(float) for k, v in out.items()}
        self.last_goal = out["q_des"]
        return out


def load_learned_interface(path, profile: ServoProfile, kp_nominal, kd_nominal, nominal=None) -> LearnedInterface:
    pol, _ = load_interface(path)
    return LearnedInterface(pol, profile, kp_nominal, kd_nominal)
