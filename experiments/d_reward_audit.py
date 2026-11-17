"""Mini reward audit (ADR 0007): per-term step reward of fixed controllers on windows of the surface
families, under the D3 reward and under the cap reward with the contact-maintenance terms.

It answers two questions before any training: where does the reward disagree with the protocol
(a controller the protocol ranks worse scores higher), and do the new terms rank a steady, compliant
press above the chattering residual and yielding above pushing. The controllers are the tuned PD
(zero residual), a soft PD (the same gains scaled down, a stand-in for a compliant controller) and
named checkpoints; the windows are the contact-focused training windows (surface families, mixed
starts), identical for every controller.

    uv run python experiments/d_reward_audit.py --out runs/d6/reward_audit.json \
        residual=runs/d4/pd_residual_acmed_s02_lowrate-s0/best.pt ppo_contact=runs/d5/ppo_contact_v2-s0/final.pt
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np
import torch

from experiments.d_reward_terms import reward_terms
from llc.controllers.pd import home_inertia_diag
from llc.dynamics.rigid_body import RigidBodyModel
from llc.learning.actor_critic import Critic, RewardConfig, rollout_ac
from llc.learning.data import reference_pool_v2, sample_batch_v2
from llc.learning.mjgrad import MjBatch
from llc.learning.policy import Policy, PolicyConfig, load_policy
from llc.sim.instances import sample_instance
from llc.sim.nominal import NominalModel
from llc.sim.panda import load_model


def contact_terms(rew: RewardConfig, res: dict, priv: dict, q_ref) -> dict[str, torch.Tensor]:
    """The contact-maintenance terms (B, T-1) from the rollout logs, aligned like ``reward_terms``
    (logged states are the states after each step, hence reference index k+1)."""
    tool = res["tool_pos"][:, :-1].float()
    touch = res["touch"]
    if getattr(rew, "force_tau", 0.0) > 0:
        a, f, acc = min(1.0, 1e-3 / rew.force_tau), torch.zeros_like(touch), torch.zeros(touch.shape[0])
        for k in range(touch.shape[1]):
            acc = acc + a * (touch[:, k] - acc); f[:, k] = acc
        touch = f
    touch = touch[:, :-1]
    e = q_ref[:, 1:] - res["q"][:, :-1]
    tool_ref, depth = torch.as_tensor(priv["tool_ref"])[:, 1:], torch.as_tensor(priv["depth_ref"])[:, 1:]
    normal, has = torch.as_tensor(priv["normal"])[:, None, :], torch.as_tensor(priv["has_surface"])[:, None]
    inside = (has & (depth > 0.0)).float()
    e_tool = tool_ref - tool
    e_tan = e_tool - normal * (normal * e_tool).sum(-1, keepdim=True)
    k_tan = torch.exp(-(e_tan * e_tan).sum(-1) / rew.feas_sigma**2)
    f_target = torch.clamp(rew.press_min + rew.press_stiffness * depth, max=rew.force_cap)
    k_force = torch.exp(-((touch - f_target) / rew.press_sigma).square())
    k_joint = torch.exp(-(e * e).mean(-1) / rew.pos_sigma**2)
    return {"tangential kernel (inside)": rew.w_feas * inside * k_tan, "force kernel (inside)": rew.w_press * inside * k_force,
            "joint kernel removed (inside)": -rew.w_pos * inside * k_joint}, inside, f_target


class ScheduledGainPolicy(Policy):
    """Exemplar of the behaviour the reward should prefer: the shared PD, stiff in free space and
    scaled to ``soft`` while the reference is inside the surface (privileged schedule, no learning).
    ``rollout_ac`` calls ``decode`` once per physics step, which indexes the schedule."""

    def __init__(self, cfg: PolicyConfig, schedule: torch.Tensor, soft: float):
        super().__init__(replace(cfg, kind="pd_gain_residual"))
        self.schedule, self.soft, self.k = schedule, soft, 0

    def decode(self, out):
        s = self.schedule[:, min(self.k, self.schedule.shape[1] - 1)]
        self.k += 1
        scale = 1.0 - (1.0 - self.soft) * s
        return {"kp_scale": scale[:, None].expand(-1, 7), "kd_scale": torch.sqrt(scale)[:, None].expand(-1, 7), "residual": torch.zeros(s.shape[0], 7)}


def physical_stats(res: dict, inside: torch.Tensor, dt: float) -> dict:
    """What the controller does while the reference is inside the surface."""
    touch = res["touch"][:, :-1]
    c = touch > 0.5
    m = inside > 0.5
    out = {"mean force inside (N)": float(touch[m].mean()) if m.any() else float("nan"), "contact fraction inside": float(c[m].float().mean()) if m.any() else float("nan"),
           "peak force (N)": float(touch.max()), "peak force after 50 ms (N)": float(touch[:, 50:].max())}
    # contact losses of >= 5 ms per second while inside
    losses, seconds = 0, 0.0
    for b in range(touch.shape[0]):
        mb = m[b].numpy()
        if not mb.any():
            continue
        cb = c[b].numpy() & mb
        edges = np.diff(cb.astype(int)); offs = np.where(edges == -1)[0]; ons = np.where(edges == 1)[0]
        for o in offs:
            nxt = ons[ons > o]
            gap = (nxt[0] - o) if nxt.size else (len(cb) - o)
            if gap * dt >= 0.005 and mb[o]:
                losses += 1
        seconds += float(mb.sum()) * dt
    out["losses per s inside"] = losses / seconds if seconds > 0 else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*", help="name=path pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gains", default="runs/d1/gains_shared.json")
    ap.add_argument("--reward-from", default="runs/d4/pd_residual_acmed_s02_lowrate-s0/meta.json", help="meta.json whose reward is the D3 reward")
    ap.add_argument("--batch", type=int, default=12, help="windows per surface family")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--soft-scale", type=float, default=0.3)
    ap.add_argument("--force-cap", type=float, default=15.0)
    ap.add_argument("--force-tau", type=float, default=0.02, help="s; the cap reward's force terms on the filtered force")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    inertia = home_inertia_diag(NominalModel())
    shared = json.load(open(args.gains))["spec"]
    cfg = PolicyConfig(kind="pd_residual", history=10, history_stride=10, history_dense=32, rate_hz=1000.0, latency_s=0.0,
                       inertia_diag=tuple(inertia.tolist()), kp_shared=tuple(shared["kp"]), kd_shared=tuple(shared["kd"]))
    policies = {"pd": Policy(cfg), "pd_soft": Policy(replace(cfg, kp_shared=tuple(args.soft_scale * k for k in shared["kp"]), kd_shared=tuple(np.sqrt(args.soft_scale) * k for k in shared["kd"])))}
    for item in args.checkpoints:
        name, path = item.split("=")
        policies[name], _ = load_policy(path)
    base = RewardConfig(**{k: v for k, v in json.load(open(args.reward_from))["reward"].items() if k in RewardConfig.__dataclass_fields__})
    base.future_offsets_ms = tuple(base.future_offsets_ms)
    cap = replace(base, force_soft=args.force_cap, force_cap=args.force_cap, w_press=1.0, force_tau=args.force_tau)
    rewards = {"d3": base, "cap": cap}
    gravity_model = RigidBodyModel(load_model(), dtype=torch.float32)
    pool = reference_pool_v2(48, "train")
    B, T, dt = args.batch, 2000, 1e-3
    batch = MjBatch([sample_instance(rng, payload_mass=float(rng.uniform(0, 3))) for _ in range(B)], seed=1)
    critic = Critic(cfg.obs_dim, 4)
    lim = policies["pd"].limits
    out = {"rewards": {k: v.__dict__ for k, v in rewards.items()}, "regimes": {}}
    for regime in ("contact2", "blocked"):
        q_ref, qd_ref, q0, qd0, labels, f_ext, priv = sample_batch_v2(pool, {regime: 1.0}, B, T, dt, rng, start_mode="mixed")
        payloads = np.array([float(rng.uniform(0, 3)) for _ in range(B)], np.float32)
        perts = [sample_instance(rng, payload_mass=float(payloads[i]), payload_offset=(rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.10)) for i in range(B)]
        q_ref_t, qd_ref_t = torch.as_tensor(q_ref), torch.as_tensor(qd_ref)
        rec = {"inside_fraction": float((priv["depth_ref"] > 0).mean()), "controllers": {}}
        schedule = torch.as_tensor(priv["depth_ref"] > 0, dtype=torch.float32)
        regime_policies = dict(policies)
        regime_policies["pd_soft_inside"] = ScheduledGainPolicy(cfg, schedule, args.soft_scale)  # privileged exemplar: soft only while the reference is inside
        for name, pol in regime_policies.items():
            if isinstance(pol, ScheduledGainPolicy):
                pol.k = 0
            batch.perturb(perts)
            with torch.no_grad():
                res = rollout_ac(pol, critic, batch, gravity_model, q_ref_t, qd_ref_t, q0, qd0, payloads, f_ext, base, horizon=100, train=False, bootstrap_scale=0.0)
            res["tool_pos"] = torch.as_tensor(np.asarray(res["tool_pos"]), dtype=torch.float32) if not torch.is_tensor(res["tool_pos"]) else res["tool_pos"]
            entry = {"stats": physical_stats(res, torch.as_tensor(priv["depth_ref"])[:, 1:] > 0, dt), "rewards": {}}
            for rname, rew in rewards.items():
                terms = reward_terms(rew, q_ref_t, qd_ref_t, res, lim, dt)
                if rew.w_press > 0:
                    extra, inside, _ = contact_terms(rew, res, priv, q_ref_t)
                    terms.update(extra)
                means = {k: float(v.mean()) for k, v in terms.items()}
                entry["rewards"][rname] = {"terms": means, "total": float(sum(means.values()))}
            rec["controllers"][name] = entry
            st = entry["stats"]
            print(f"{regime:9s} {name:12s} d3 {entry['rewards']['d3']['total']:8.4f} cap {entry['rewards']['cap']['total']:8.4f} | "
                  f"force inside {st['mean force inside (N)']:5.1f} N, contact {st['contact fraction inside']:.2f}, losses/s {st['losses per s inside']:5.1f}, peak {st['peak force (N)']:5.0f} N (after 50 ms {st['peak force after 50 ms (N)']:5.0f})", flush=True)
        out["regimes"][regime] = rec
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
