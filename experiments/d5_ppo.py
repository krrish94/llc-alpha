"""Deep dive D4: PPO fine-tuning of the PD + residual controller in the MuJoCo plant.

    uv run python experiments/d5_ppo.py --warmstart runs/d4/pd_residual_acmed_s02_lowrate-s0/best.pt \
        --critic runs/d4/pd_residual_acmed_s02_lowrate-s0/critic.pt --families all --out runs/d5/ppo_all-s0

The actor is the warm-start policy with a learned diagonal Gaussian in its pre-tanh output; the
critic is the privileged critic of the actor-critic run (state, touch force, payload, reference
preview). Reward, kernel width and penalty weights are read from the warm-start run's meta.json
unless overridden; the contact-force threshold is lowered and a force reflex ends the window.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from llc.dynamics.rigid_body import RigidBodyModel
from llc.eval.cases_v2 import REGIMES_V2
from llc.eval.parallel import CaseSpec, evaluate, validate_policy
from llc.learning.actor_critic import Critic, RewardConfig
from llc.learning.data import reference_pool_v2
from llc.learning.mjgrad import MjBatch
from llc.learning.policy import Policy, PolicyConfig, convert_to_gain_residual, expand_ext_torque, load_policy, save_policy
from llc.learning.ppo import GaussianActor, PPOConfig, PPOEnv, collect, ppo_update
from llc.sim.instances import sample_instance
from llc.sim.panda import load_model

FAMILIES = {
    "all": {"hold": 0.1, "free": 0.2, "policy": 0.15, "step": 0.15, "contact2": 0.2, "blocked": 0.2},
    "contact": {"hold": 0.1, "free": 0.1, "contact2": 0.4, "blocked": 0.4},
    "free": {"hold": 0.3, "free": 0.4, "policy": 0.15, "step": 0.15},
    # declared: the task family enters training (no longer held out) for the multi-task recipe
    "full": {"hold": 0.1, "free": 0.15, "policy": 0.1, "step": 0.1, "contact2": 0.2, "blocked": 0.2, "task": 0.15},
    "contact_focus": {"hold": 0.15, "free": 0.2, "policy": 0.1, "contact2": 0.25, "blocked": 0.3},
    # Milestone 12 (ADR 0012): every family of protocol v5, the contact-rich ones weighted 1.5 to 1
    "consolidation": {"hold": 0.6, "free": 0.8, "policy": 0.5, "step": 0.5, "contact2": 1.0, "blocked": 1.2, "task": 0.8, "push": 0.8, "pick": 0.8, "wipe": 1.2, "erase": 1.2, "peg": 1.5, "drawer": 1.2, "door": 1.2},  # contact deep dive (ADR 0007); the task family stays held out; free-space share keeps the tracking from being forgotten
}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def reward_from_run(run_dir: Path) -> RewardConfig:
    """The reward of a run's meta.json; a run without one (a distilled student) inherits its own warm start's."""
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if "reward" in meta:
        return RewardConfig(**{k: v for k, v in meta["reward"].items() if k in RewardConfig.__dataclass_fields__})
    ws = meta.get("args", {}).get("warmstart")
    return reward_from_run(Path(ws).parent) if ws else RewardConfig()


def validate(ckpt: Path, n: int, n_contact: int | None = None, objective: str = "composite") -> dict:
    return validate_policy(ckpt, n, n if n_contact is None else n_contact, objective)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", required=True, help="pd_residual policy checkpoint (best.pt of an actor-critic run)")
    ap.add_argument("--critic", default="", help="critic.pt of the same run (privileged critic warm start)")
    ap.add_argument("--scratch", action="store_true", help="same architecture and reward, but fresh weights (zero residual) and a fresh critic")
    ap.add_argument("--families", default="all", choices=sorted(FAMILIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="")
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--hold", type=int, default=10, help="physics steps per decision")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--lr-actor", type=float, default=1e-4)
    ap.add_argument("--lr-critic", type=float, default=3e-4)
    ap.add_argument("--init-std", type=float, default=0.1)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gamma-step", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--reflex-force", type=float, default=40.0)
    ap.add_argument("--reflex-penalty", type=float, default=5.0)
    ap.add_argument("--reflex-tau", type=float, default=0.0, help="s; > 0 applies the reflex to a low-pass filtered (sustained) force")
    ap.add_argument("--feasible", action="store_true", help="privileged feasible-direction tracking kernel in contact (ADR 0006)")
    ap.add_argument("--no-reflex-terminate", action="store_true", help="charge the reflex penalty at each crossing but keep the window (train post-impact behaviour)")
    ap.add_argument("--reg-warmstart", type=float, default=0.0, help="L2 penalty weight between the actor mean and the frozen warm-start mean")
    ap.add_argument("--feas-sigma", type=float, default=None)
    ap.add_argument("--feas-threshold", type=float, default=1.0, help="N; the normal component of the tool error is rewarded until the contact force exceeds this (30: press toward the target up to the cap)")
    ap.add_argument("--force-soft", type=float, default=None, help="contact force above which the penalty applies (N; default: the cap if given, else 20)")
    # contact deep dive (ADR 0007)
    ap.add_argument("--gain-residual", action="store_true", help="convert the warm start to pd_gain_residual (gain multipliers + residual)")
    ap.add_argument("--ext-torque", action="store_true", help="add the external-torque estimate to the history (warm start expanded exactly)")
    ap.add_argument("--output-tau", type=float, default=0.0, help="s; first-order low-pass of the held output at the physics rate")
    ap.add_argument("--start-mode", default="contact", choices=["contact", "mixed"])
    ap.add_argument("--noise-rho", type=float, default=0.0, help="AR(1) correlation of the exploration noise between decisions")
    ap.add_argument("--force-tau", type=float, default=None, help="s; force penalties on the low-pass filtered contact force (0 = instantaneous)")
    ap.add_argument("--w-press", type=float, default=0.0, help="weight of the contact-maintenance force kernel (0 = off)")
    ap.add_argument("--press-sigma", type=float, default=None)
    ap.add_argument("--press-stiffness", type=float, default=None)
    ap.add_argument("--press-min", type=float, default=None)
    ap.add_argument("--force-cap", type=float, default=None, help="N; force kernel cap and (unless --force-soft) the penalty threshold")
    ap.add_argument("--w-feas", type=float, default=None)
    ap.add_argument("--val-cases-contact", type=int, default=None)
    ap.add_argument("--val-objective", default="composite", choices=["composite", "composite_cap"])
    ap.add_argument("--w-force", type=float, default=None)
    ap.add_argument("--pos-sigma", type=float, default=None)
    ap.add_argument("--val-every", type=int, default=20)
    ap.add_argument("--val-cases", type=int, default=2)
    ap.add_argument("--pool", type=int, default=48)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sim-threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # warm start: same network, decision period set to the PPO hold
    src_policy, src_extra = load_policy(args.warmstart)
    if args.gain_residual and src_policy.cfg.kind == "pd_residual":
        src_policy = convert_to_gain_residual(src_policy)
    if args.ext_torque and not src_policy.cfg.ext_torque:
        src_policy = expand_ext_torque(src_policy)
    assert src_policy.cfg.kind in ("pd_residual", "pd_gain_residual"), "PPO fine-tuning is implemented for the residual controllers"
    cfg = replace(src_policy.cfg, hold_steps=args.hold, output_tau=args.output_tau)
    policy = Policy(cfg)
    if not args.scratch:
        policy.load_state_dict(src_policy.state_dict())
    rew = reward_from_run(Path(args.warmstart).parent)
    rew.future_offsets_ms = tuple(rew.future_offsets_ms)
    if args.force_cap is not None:
        rew.force_cap = args.force_cap
    rew.force_soft = args.force_soft if args.force_soft is not None else (args.force_cap if args.force_cap is not None else 20.0)
    rew.w_press = args.w_press
    if args.force_tau is not None:
        rew.force_tau = args.force_tau
    for name in ("press_sigma", "press_stiffness", "press_min", "w_feas"):
        if getattr(args, name) is not None:
            setattr(rew, name, getattr(args, name))
    if args.feas_sigma is not None:
        rew.feas_sigma = args.feas_sigma
    if args.w_force is not None:
        rew.w_force = args.w_force
    if args.pos_sigma is not None:
        rew.pos_sigma = args.pos_sigma
    pcfg = PPOConfig(hold_steps=args.hold, gamma_step=args.gamma_step, lam=args.lam, clip=args.clip, epochs=args.epochs, minibatch=args.minibatch,
                     lr_actor=args.lr_actor, lr_critic=args.lr_critic, init_std=args.init_std, entropy=args.entropy,
                     reflex_force=args.reflex_force, reflex_penalty=args.reflex_penalty, reflex_tau=args.reflex_tau,
                     feasible_kernel=args.feasible, feas_sigma=args.feas_sigma if args.feas_sigma is not None else 0.01, contact_threshold=args.feas_threshold, reflex_terminate=not args.no_reflex_terminate, reg_warmstart=args.reg_warmstart,
                     start_mode=args.start_mode, noise_rho=args.noise_rho)
    reference = None
    if args.reg_warmstart > 0 and not args.scratch:
        reference = Policy(cfg); reference.load_state_dict(src_policy.state_dict()); reference.eval()
    actor = GaussianActor(policy, pcfg.init_std, reference)
    critic = Critic(cfg.obs_dim, len(rew.future_offsets_ms))
    if args.critic and not args.scratch:
        critic.load_state_dict(torch.load(args.critic, map_location="cpu"))
    opt_actor = torch.optim.Adam(actor.parameters(), lr=pcfg.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=pcfg.lr_critic)
    gravity_model = RigidBodyModel(load_model(), dtype=torch.float32)
    pool = reference_pool_v2(args.pool, "train", regimes=tuple(FAMILIES[args.families]))
    horizon_steps = int(round(args.horizon / 1e-3))
    batch = MjBatch([sample_instance(rng, payload_mass=float(rng.uniform(0, 3))) for _ in range(args.envs)], seed=args.seed, threads=args.sim_threads)
    env = PPOEnv(batch, policy, gravity_model, pool, FAMILIES[args.families], horizon_steps, rng, rew, pcfg)
    meta = {"args": vars(args), "method_key": f"ppo_{args.variant or args.families}", "policy_config": cfg.__dict__, "reward": rew.__dict__, "ppo": pcfg.__dict__, "families": FAMILIES[args.families],
            "warmstart_extra": src_extra, "git": git_hash(), "protocol": "v2", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "env_steps_per_update": args.envs * env.decisions, "physics_steps_per_update": args.envs * horizon_steps}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log_f = open(out / "log.jsonl", "a")
    best = {"val": float("inf"), "update": -1}
    t0 = time.time()
    # validation of the warm start itself under the decision period (update -1)
    save_policy(out / "last.pt", policy, {"update": -1})
    val = validate(out / "last.pt", args.val_cases, args.val_cases_contact, args.val_objective)
    log_f.write(json.dumps({"update": -1, "val": val}) + "\n"); log_f.flush()
    print(f"[ppo {args.families} s{args.seed}] warm start under hold {args.hold}: val {val['mean']:.3f} exc {val['excess_force']:.1f} over40N {val['over_reflex']} sustained {val['over_reflex_20ms']} " + " ".join(f"{k[:4]}={v:.2f}" for k, v in val["per_regime"].items()), flush=True)
    best = {"val": val["mean"], "update": -1}
    save_policy(out / "best.pt", policy, {"update": -1, "val": val})
    for update in range(args.updates):
        roll = collect(env, actor, critic)
        stats = ppo_update(actor, critic, opt_actor, opt_critic, roll, pcfg)
        labels = np.asarray(roll["labels"])
        ret_per_env = roll["rew"].sum(0)
        rec = {"update": update, "env_steps": (update + 1) * args.envs * env.decisions, "mean_return": float(ret_per_env.mean()),
               "return_by_family": {l: float(ret_per_env[labels == l].mean()) for l in sorted(set(labels))},
               "reflex_rate": float(roll["reflex"].float().mean()), "max_touch": float(roll["max_touch"].max()), "std": float(actor.log_std.exp().mean()),
               **stats, "elapsed_s": time.time() - t0}
        if (update + 1) % args.val_every == 0 or update + 1 == args.updates:
            save_policy(out / "last.pt", policy, {"update": update})
            save_policy(out / f"ckpt_{update:04d}.pt", policy, {"update": update})
            torch.save(critic.state_dict(), out / "critic.pt")
            val = validate(out / "last.pt", args.val_cases, args.val_cases_contact, args.val_objective)
            rec["val"] = val
            if val["mean"] < best["val"]:
                best = {"val": val["mean"], "update": update}
                save_policy(out / "best.pt", policy, {"update": update, "val": val})
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        msg = (f"[ppo {args.families} s{args.seed}] upd {update:3d} return {rec['mean_return']:.2f} reflex {rec['reflex_rate']:.2f} maxF {rec['max_touch']:.0f} std {rec['std']:.3f} "
               f"kl {rec['approx_kl']:.4f} clip {rec['clip_frac']:.2f} vloss {rec['value_loss']:.3f} ({rec['elapsed_s']:.0f}s)")
        if "val" in rec:
            msg += f" | val {val['mean']:.3f} exc {val['excess_force']:.1f} over40N {val['over_reflex']} sustained {val['over_reflex_20ms']} " + " ".join(f"{k[:4]}={v:.2f}" for k, v in val["per_regime"].items())
        print(msg, flush=True)
    save_policy(out / "final.pt", policy, {"update": args.updates - 1})
    torch.save(critic.state_dict(), out / "critic.pt")
    (out / "best.json").write_text(json.dumps(best))
    print(f"done; best val {best['val']:.3f} at update {best['update']}; wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
