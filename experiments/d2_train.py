"""Deep dive (ADR 0005): train PD + residual or neural torque at 1 kHz through MuJoCo finite-difference
gradients, with per-window robot instances and payloads, the protocol v2 reference families, and
the contact-force loss terms switchable for the compliance ablation.

    uv run python experiments/d2_train.py --kind pd_residual --seed 0 --out runs/d2/pd_residual-s0
    uv run python experiments/d2_train.py --kind neural_torque --seed 0 --no-force --out runs/d2/neural_torque_noforce-s0
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

from llc.controllers.pd import home_inertia_diag
from llc.dynamics.rigid_body import RigidBodyModel
from llc.eval.cases_v2 import REGIMES_V2
from llc.eval.parallel import CaseSpec, evaluate, validate_policy
from llc.learning.actor_critic import Critic, RewardConfig, fit_critic, rollout_ac, td_lambda_targets
from llc.learning.data import reference_pool_v2, sample_batch_v2
from llc.learning.decoupled import DecoupledBatch, DynamicsNet, ImaginedBatch
from llc.learning.mjgrad import MjBatch
from llc.learning.policy import HistoryBuffer, Policy, PolicyConfig, convert_to_gain_residual, expand_ext_torque, load_policy, save_policy
from llc.learning.rollout import LossConfig
from llc.learning.rollout_mj import rollout_mj
from llc.sim.instances import sample_instance
from llc.sim.nominal import NominalModel
from llc.sim.panda import NUM_JOINTS, load_model

STAGES = [
    {"hold": 0.3, "free": 0.5, "policy": 0.2},
    {"hold": 0.15, "free": 0.3, "policy": 0.25, "step": 0.3},
    {"hold": 0.1, "free": 0.2, "policy": 0.15, "step": 0.15, "contact2": 0.2, "blocked": 0.2},
]
CONTACT_MIX = {"hold": 0.15, "free": 0.2, "policy": 0.1, "contact2": 0.25, "blocked": 0.3}  # contact deep dive (ADR 0007); the task family stays held out; free-space share keeps the tracking from being forgotten
MIXES = {"stages": STAGES, "contact": [CONTACT_MIX] * 3}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def validate(ckpt: Path, n: int, n_contact: int | None = None, objective: str = "composite") -> dict:
    return validate_policy(ckpt, n, n if n_contact is None else n_contact, objective)


def warm_start(policy: Policy, pool: dict, steps: int, rng: np.random.Generator, gains: dict, horizon_steps: int, feedforward: bool = False) -> None:
    """Imitate a PD + gravity controller (plus the inverse-dynamics feedforward when ``feedforward``) on
    closed-loop free-space states produced in the analytic model at 1 kHz with the policy's own history
    (neural torque only; disclosed)."""
    from llc.learning.actor_critic import reference_accel

    model = RigidBodyModel(load_model(), dtype=torch.float32)
    kp, kd = torch.tensor(gains["kp"], dtype=torch.float32), torch.tensor(gains["kd"], dtype=torch.float32)
    lim = policy.limits
    obs_all, tgt_all = [], []
    with torch.no_grad():
        for _ in range(4):
            q_ref, qd_ref, q0, qd0, _, _, _priv = sample_batch_v2(pool, {"hold": 0.3, "free": 0.5, "policy": 0.2}, 32, horizon_steps, 1e-3, rng, init_noise_q=0.05, pulse_prob=0.0)
            q_ref, qd_ref = torch.as_tensor(q_ref), torch.as_tensor(qd_ref)
            qdd_ref = reference_accel(qd_ref, 1e-3)
            q, qd = torch.as_tensor(q0), torch.as_tensor(qd0)
            buf = HistoryBuffer(policy.cfg, q, qd, model.inverse_dynamics if policy.cfg.ext_torque else None, 1e-3)
            last_tau = torch.zeros(32, NUM_JOINTS)
            for k in range(q_ref.shape[1]):
                buf.push(q, qd, last_tau, q_ref[:, k])
                g = model.inverse_dynamics(q, qd_ref[:, k], qdd_ref[:, k]) if feedforward else model.bias(q, torch.zeros_like(q))
                tau = torch.maximum(torch.minimum(kp * (q_ref[:, k] - q) + kd * (qd_ref[:, k] - qd) + g, lim), -lim)
                if k % 4 == 0:
                    obs_all.append(buf.observation(q, qd, q_ref[:, k], qd_ref[:, k], lim))
                    tgt_all.append(tau)
                q, qd = model.step(q, qd, tau, 1e-3)
                last_tau = tau
    obs_all, tgt_all = torch.cat(obs_all), torch.cat(tgt_all)
    opt = torch.optim.Adam(policy.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    n = len(obs_all)
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, 512))
        loss = ((policy.decode(policy(obs_all[idx]))["torque"] - tgt_all[idx]) / lim).square().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"[warmstart] step {step} torque mse (normalized) {loss.item():.5f} on {n} samples", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["pd_residual", "neural_torque", "pd_gain_residual"])
    # contact deep dive (ADR 0007)
    ap.add_argument("--init-from", default="", help="pd_residual checkpoint to start from (converted to pd_gain_residual / expanded with the external-torque feature as needed); its critic.pt is loaded when present")
    ap.add_argument("--mix", default="stages", choices=sorted(MIXES), help="training mixture: the D2/D3 curriculum or the contact-focused mixture in every stage")
    ap.add_argument("--start-mode", default="contact", choices=["contact", "mixed"], help="mixed: half of the surface windows start before the release (recovery trained)")
    ap.add_argument("--force-tau", type=float, default=None, help="s; force penalties on the low-pass filtered contact force (0 = instantaneous)")
    ap.add_argument("--w-press", type=float, default=0.0, help="weight of the contact-maintenance force kernel (0 = D3 reward)")
    ap.add_argument("--press-sigma", type=float, default=None)
    ap.add_argument("--press-stiffness", type=float, default=None)
    ap.add_argument("--press-min", type=float, default=None)
    ap.add_argument("--force-cap", type=float, default=None, help="N; sets the force kernel cap and, unless --force-soft is given, the force penalty threshold")
    ap.add_argument("--w-feas", type=float, default=None)
    ap.add_argument("--feas-sigma", type=float, default=None)
    ap.add_argument("--ext-torque", action="store_true", help="external-torque estimate from the robot model in the history")
    ap.add_argument("--feedforward", action="store_true", help="PD base with the nominal inverse-dynamics feedforward of the reference (ADR 0010 round 2)")
    ap.add_argument("--layers", type=int, default=2, help="trunk depth (ADR 0012: 4 and 10)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--residual-blocks", action="store_true", help="residual trunk with layer normalisation (ADR 0012, the 10-layer student)")
    ap.add_argument("--output-tau", type=float, default=0.0, help="s; first-order low-pass of the held output at the physics rate")
    ap.add_argument("--residual-fraction", type=float, default=0.5)
    ap.add_argument("--val-cases-contact", type=int, default=None, help="validation cases per surface family (default: --val-cases)")
    ap.add_argument("--val-objective", default="composite", choices=["composite", "composite_cap", "free"], help="free: the cap composite over the free-space families only (the base of a distilled student, ADR 0010 round 2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-force", action="store_true", help="drop the contact-force loss terms (compliance ablation)")
    ap.add_argument("--force-scale", type=float, default=1.0, help="multiplies the force and force-rate loss weights")
    ap.add_argument("--force-grad", default="full", choices=["full", "local", "detach"], help="local: the touch-sensor gradient flows only through the control input, not the state")
    ap.add_argument("--objective", default="loss", choices=["loss", "ac"], help="ac: short-horizon actor-critic with bounded tracking kernels (gradient plan, item 1)")
    ap.add_argument("--ac-horizon", type=int, default=100)
    ap.add_argument("--median-clip", type=float, default=0.0, help=">0: per-sample state-gradient clip at this multiple of the batch median (item 3)")
    ap.add_argument("--pos-sigma", type=float, default=0.05)
    ap.add_argument("--w-rate", type=float, default=None, help="torque-rate penalty weight of the actor-critic reward (default RewardConfig.w_rate)")
    ap.add_argument("--w-impulse", type=float, default=None, help="contact force-rate penalty weight (default RewardConfig.w_impulse)")
    ap.add_argument("--vel-sigma", type=float, default=None, help="velocity kernel half-width in rad/s (default RewardConfig.vel_sigma)")
    ap.add_argument("--w-lin", type=float, default=None, help="weight of the linear restoring term (default RewardConfig.w_lin; 0 removes it)")
    ap.add_argument("--force-soft", type=float, default=None, help="contact force above which the force penalty applies, N (default RewardConfig.force_soft)")
    ap.add_argument("--ref-history", action="store_true", help="add the position-target history (same windows) to the observation")
    ap.add_argument("--critic-lr", type=float, default=5e-4)
    ap.add_argument("--no-critic", action="store_true", help="actor-critic objective without the bootstrap (myopic chunk returns; SuperTrack-style fallback)")
    ap.add_argument("--grad", default="fd", choices=["fd", "learned", "imagined"], help="learned: decoupled gradients through an online-fitted dynamics model (item 2); imagined: forward and backward in the model, plant rollouts only to refit it (item 5)")
    ap.add_argument("--imagine-updates", type=int, default=4, help="imagined policy updates per plant data-collection rollout (imagined mode)")
    ap.add_argument("--model-warmup", type=int, default=15, help="rollouts collected and fitted before policy updates start (learned gradients)")
    ap.add_argument("--model-batches", type=int, default=10)
    ap.add_argument("--stage-iters", type=int, nargs=3, default=[150, 150, 400])
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--tbptt", type=int, default=100)
    ap.add_argument("--state-grad-clip", type=float, default=1e3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-cases", type=int, default=2)
    ap.add_argument("--pool", type=int, default=48)
    ap.add_argument("--history", type=int, default=10)
    ap.add_argument("--history-stride", type=int, default=10)
    ap.add_argument("--history-dense", type=int, default=32)
    ap.add_argument("--encoding", default="raw")
    ap.add_argument("--warmstart", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sim-threads", type=int, default=None, help="simulator pool size of the finite-difference batch (default: MjBatch default)")
    ap.add_argument("--gains", default="runs/d1/gains_shared.json", help="protocol v2 shared PD gains (residual base)")
    ap.add_argument("--variant", default="")
    ap.add_argument("--keep-checkpoints", action="store_true", help="also keep every validated checkpoint as ckpt_<iter>.pt (post-hoc selection)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    nominal = NominalModel()
    inertia = home_inertia_diag(nominal)
    shared = json.load(open(args.gains))["spec"]
    cfg = PolicyConfig(kind=args.kind, history=args.history, history_stride=args.history_stride, history_dense=args.history_dense, rate_hz=1000.0, latency_s=0.0,
                       encoding=args.encoding, inertia_diag=tuple(inertia.tolist()), kp_shared=tuple(shared["kp"]), kd_shared=tuple(shared["kd"]), ref_history=args.ref_history,
                       ext_torque=args.ext_torque, output_tau=args.output_tau, residual_fraction=args.residual_fraction, feedforward=args.feedforward,
                       layers=args.layers, hidden=args.hidden, residual_blocks=args.residual_blocks)
    policy = Policy(cfg)
    stages = MIXES[args.mix]
    init_critic = None
    if args.init_from:
        src, _ = load_policy(args.init_from)
        if args.kind == "pd_gain_residual" and src.cfg.kind == "pd_residual":
            src = convert_to_gain_residual(src)
        if args.ext_torque and not src.cfg.ext_torque:
            src = expand_ext_torque(src)
        assert src.cfg.kind == args.kind, (src.cfg.kind, args.kind)
        cfg = replace(src.cfg, output_tau=args.output_tau)
        policy = Policy(cfg); policy.load_state_dict(src.state_dict()); policy.train()
        if (Path(args.init_from).parent / "critic.pt").exists():
            init_critic = Path(args.init_from).parent / "critic.pt"
        print(f"[init] from {args.init_from}: kind {cfg.kind}, obs {cfg.obs_dim}, ext_torque {cfg.ext_torque}, output_tau {cfg.output_tau}", flush=True)
    gravity_model = RigidBodyModel(load_model(), dtype=torch.float32)
    lcfg = LossConfig()
    if args.no_force:
        lcfg.force_weight, lcfg.impulse_weight = 0.0, 0.0
    lcfg.force_weight *= args.force_scale
    lcfg.impulse_weight *= args.force_scale
    horizon_steps = int(round(args.horizon / 1e-3))
    pool = reference_pool_v2(args.pool, "train")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    rew = RewardConfig(pos_sigma=args.pos_sigma)
    if args.w_rate is not None:
        rew.w_rate = args.w_rate
    if args.w_impulse is not None:
        rew.w_impulse = args.w_impulse
    if args.vel_sigma is not None:
        rew.vel_sigma = args.vel_sigma
    if args.w_lin is not None:
        rew.w_lin = args.w_lin
    if args.force_cap is not None:
        rew.force_cap = args.force_cap
        rew.force_soft = args.force_cap
    if args.force_soft is not None:
        rew.force_soft = args.force_soft
    rew.w_press = args.w_press
    if args.force_tau is not None:
        rew.force_tau = args.force_tau
    for name in ("press_sigma", "press_stiffness", "press_min", "w_feas", "feas_sigma"):
        if getattr(args, name) is not None:
            setattr(rew, name, getattr(args, name))
    if args.no_force:
        rew.w_force, rew.w_impulse = 0.0, 0.0
    rew.w_force *= args.force_scale; rew.w_impulse *= args.force_scale
    critic = Critic(cfg.obs_dim, len(rew.future_offsets_ms))
    if init_critic is not None:
        critic.load_state_dict(torch.load(init_critic, map_location="cpu"))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    total = sum(args.stage_iters)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total, eta_min=args.lr * 0.05)
    meta = {"args": vars(args), "policy_config": cfg.__dict__, "loss": lcfg.__dict__, "reward": rew.__dict__, "stages": stages, "git": git_hash(), "protocol": "v2",
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log_f = open(out / "log.jsonl", "a")
    if args.kind == "neural_torque" and args.warmstart > 0:  # imitate the tuned PD (with the feedforward when asked), not the tuning's initial guess
        warm_start(policy, pool, args.warmstart, rng, shared, horizon_steps, feedforward=args.feedforward)
    batch = MjBatch([sample_instance(rng, payload_mass=float(rng.uniform(0, 3))) for _ in range(args.batch)], seed=args.seed, threads=args.sim_threads)
    batch.touch_grad_local = args.force_grad == "local"
    if args.force_grad == "detach":
        rew.force_grad = "detach"
    payloads = np.zeros(args.batch, np.float32)
    plant = None
    if args.grad in ("learned", "imagined"):
        batch.perturbations = [sample_instance(rng, payload_mass=float(rng.uniform(0, 3))) for _ in range(args.batch)]
        batch = DecoupledBatch(batch, DynamicsNet(), rng)
        if args.grad == "imagined":
            plant = batch  # keeps collecting plant transitions and fitting the model
            batch = ImaginedBatch(plant.net, args.batch)
        # model warm-up: collect transitions with the initial policy and fit before the first policy update
        collector = plant if plant is not None else batch
        for w in range(args.model_warmup):
            q_ref, qd_ref, q0, qd0, labels, f_ext, _priv = sample_batch_v2(pool, stages[2], args.batch, horizon_steps, 1e-3, rng)
            collector.perturb([sample_instance(rng, payload_mass=float(rng.uniform(0, 3))) for _ in range(args.batch)])
            with torch.no_grad():
                rollout_mj(policy, collector, gravity_model, torch.as_tensor(q_ref), torch.as_tensor(qd_ref), q0, qd0, f_ext, None, train=False, tbptt_steps=0, obs_noise_q=1e-3)
            fit = collector.fit(batches=200)
            print(f"[model warm-up] rollout {w} buffer {collector.replay.n} loss {fit.get('model_loss', float('nan')):.4f}", flush=True)
    best = {"val": float("inf"), "iter": -1}
    it, t0 = 0, time.time()
    for stage, n_iters in enumerate(args.stage_iters):
        for _ in range(n_iters):
            q_ref, qd_ref, q0, qd0, labels, f_ext, _priv = sample_batch_v2(pool, stages[stage], args.batch, horizon_steps, 1e-3, rng, start_mode=args.start_mode)
            payloads = np.array([float(rng.uniform(0, 3)) for _ in range(args.batch)], np.float32)
            batch.perturb([sample_instance(rng, payload_mass=float(payloads[i]), payload_offset=(rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.10)) for i in range(args.batch)])
            opt.zero_grad()
            if args.objective == "ac":
                res = rollout_ac(policy, critic, batch, gravity_model, torch.as_tensor(q_ref), torch.as_tensor(qd_ref), q0, qd0, payloads, f_ext, rew,
                                 horizon=args.ac_horizon, train=True, state_grad_clip=args.state_grad_clip, median_clip=args.median_clip, bootstrap_scale=0.0 if args.no_critic else min(1.0, it / 50.0), priv=_priv)
                gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
                opt.step(); sched.step()
                targets = td_lambda_targets(critic, res["critic_inputs"], res["rewards"], rew.gamma, rew.lam)
                closs = fit_critic(critic, critic_opt, res["critic_inputs"], targets)
                res["loss"] = -res["stats"]["reward"]
                res["loss_terms"] = {"reward": res["stats"]["reward"], "kernel": res["stats"]["kernel_pos"], "force": res["stats"]["force_pen"], "actor": res["stats"]["actor_loss"], "critic": closs}
            else:
                res = rollout_mj(policy, batch, gravity_model, torch.as_tensor(q_ref), torch.as_tensor(qd_ref), q0, qd0, f_ext, lcfg, train=True, tbptt_steps=args.tbptt, state_grad_clip=args.state_grad_clip)
                gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
                opt.step(); sched.step()
            if args.grad == "learned":
                res["loss_terms"].update(batch.fit(batches=args.model_batches))
            elif args.grad == "imagined":
                if it % args.imagine_updates == 0:  # collect fresh plant data with the current policy (action noise widens the fit)
                    with torch.no_grad():
                        plant.perturb([sample_instance(rng, payload_mass=float(rng.uniform(0, 3))) for _ in range(args.batch)])
                        rollout_mj(policy, plant, gravity_model, torch.as_tensor(q_ref), torch.as_tensor(qd_ref), q0, qd0, f_ext, None, train=False, tbptt_steps=0, obs_noise_q=1e-3)
                res["loss_terms"].update(plant.fit(batches=args.model_batches))
            rec = {"iter": it, "stage": stage, "loss": res["loss"], "terms": res["loss_terms"], "grad_norm": gn, "max_touch": float(res["touch"].max()), "elapsed_s": time.time() - t0}
            if (it + 1) % args.val_every == 0 or it + 1 == total:
                save_policy(out / "last.pt", policy, {"iter": it, "stage": stage})
                if args.keep_checkpoints:
                    save_policy(out / f"ckpt_{it:04d}.pt", policy, {"iter": it, "stage": stage})
                if args.grad == "learned":  # keep the current dynamics model with the checkpoint (an interrupted run still has one)
                    torch.save(batch.net.state_dict(), out / "dynamics_model.pt")
                elif args.grad == "imagined":
                    torch.save(plant.net.state_dict(), out / "dynamics_model.pt")
                val = validate(out / "last.pt", args.val_cases, args.val_cases_contact, args.val_objective)
                rec["val"] = val
                if val["mean"] < best["val"]:
                    best = {"val": val["mean"], "iter": it}
                    save_policy(out / "best.pt", policy, {"iter": it, "stage": stage, "val": val})
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            msg = f"[{args.kind}{'-nf' if args.no_force else ''} s{args.seed}] it {it:4d} st {stage} loss {res['loss']:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in res["loss_terms"].items()) + f" gn {gn:.2f} maxF {rec['max_touch']:.0f} ({rec['elapsed_s']:.0f}s)"
            if "val" in rec:
                msg += f" | val {val['mean']:.3f} exc {val['excess_force']:.1f} " + " ".join(f"{k[:4]}={v:.2f}" for k, v in val["per_regime"].items())
            print(msg, flush=True)
            it += 1
    save_policy(out / "final.pt", policy, {"iter": it - 1, "stage": 2})
    if args.objective == "ac":
        torch.save(critic.state_dict(), out / "critic.pt")
    if args.grad == "learned":
        torch.save(batch.net.state_dict(), out / "dynamics_model.pt")
    elif args.grad == "imagined":
        torch.save(plant.net.state_dict(), out / "dynamics_model.pt")
    (out / "best.json").write_text(json.dumps(best))
    print(f"done; best val {best['val']:.3f} at iter {best['iter']}; wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
