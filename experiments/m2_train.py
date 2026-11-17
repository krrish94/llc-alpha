"""Milestone 2: train a learned low-level controller by backpropagation through the known
differentiable model (torch rigid-body dynamics + penalty contact).

    uv run python experiments/m2_train.py --kind learned_gains --seed 0 --out runs/m2/learned_gains-s0

Curriculum (protocol v1): stage 1 hold + smooth, stage 2 adds zoh + step, stage 3 adds contact +
transition. Validation every ``--val-every`` iterations on the 'val' split in MuJoCo (the plant);
the checkpoint with the best mean validation composite is kept as best.pt, the last as final.pt.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from llc.controllers.pd import home_inertia_diag
from llc.dynamics.contact import ContactParams
from llc.dynamics.rigid_body import RigidBodyModel
from llc.eval.cases import REGIMES
from llc.eval.parallel import CaseSpec, evaluate
from llc.learning.data import reference_pool, sample_batch
from llc.learning.policy import Policy, PolicyConfig, save_policy
from llc.learning.randomize import ModelRandomizer, RandomizationRanges
from llc.learning.rollout import LossConfig, RigidBodyDynamics, RolloutConfig, rollout
from llc.sim.nominal import NominalModel
from llc.sim.panda import load_model

STAGES = [
    {"hold": 0.5, "smooth": 0.5},
    {"hold": 0.2, "smooth": 0.3, "zoh": 0.25, "step": 0.25},
    {"hold": 0.1, "smooth": 0.2, "zoh": 0.15, "step": 0.15, "contact": 0.2, "transition": 0.2},
]
CONTACT = ContactParams(k_n=40000.0, c_n=400.0, mu=0.5, eps=1e-3)  # calibrated in experiments/m2_calibrate_contact.py (peak and steady forces within 10% of MuJoCo)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def warm_start(policy: Policy, pool: dict, rcfg: RolloutConfig, steps: int, rng: np.random.Generator, inertia, shared: dict) -> None:
    """Supervised initialization of the neural-torque policy: imitate the shared PD + gravity torque
    on states and histories produced by that PD controller itself in the torch model (closed loop,
    same update rate, latency and history as the policy). A zero-torque initial policy lets the arm
    collapse and gives meaningless early gradients. This is disclosed in the report as the
    initialization; it gives the policy no information the PD baseline lacks."""
    from llc.learning.policy import encode_obs

    model = RigidBodyModel(load_model(), dtype=torch.float32)
    kp, kd = torch.tensor(shared["kp"], dtype=torch.float32), torch.tensor(shared["kd"], dtype=torch.float32)
    lim = policy.limits
    period = int(round(1.0 / (policy.cfg.rate_hz * rcfg.dt)))
    H = policy.cfg.history
    obs_all, tgt_all = [], []
    with torch.no_grad():
        for _ in range(6):  # 6 batches of 32 windows of 1.5 s
            q_ref, qd_ref, q0, qd0, _ = sample_batch(pool, {r: 1.0 for r in pool}, 32, int(1.5 / rcfg.dt), rcfg.dt, rng, init_noise_q=0.05)
            q_ref, qd_ref = torch.as_tensor(q_ref), torch.as_tensor(qd_ref)
            q, qd = torch.as_tensor(q0), torch.as_tensor(qd0)
            hist_q, hist_qd, hist_tau = q[:, None].repeat(1, H, 1), qd[:, None].repeat(1, H, 1), torch.zeros(32, H, 7)
            last_tau = torch.zeros(32, 7)
            for k in range(q_ref.shape[1]):
                g = model.bias(q, torch.zeros_like(q))
                tau = kp * (q_ref[:, k] - q) + kd * (qd_ref[:, k] - qd) + g
                tau = torch.maximum(torch.minimum(tau, lim), -lim)
                if k % period == 0:
                    hist_q = torch.cat([hist_q[:, 1:], q[:, None]], 1)
                    hist_qd = torch.cat([hist_qd[:, 1:], qd[:, None]], 1)
                    hist_tau = torch.cat([hist_tau[:, 1:], last_tau[:, None]], 1)
                    obs_all.append(encode_obs(q, qd, q_ref[:, k], qd_ref[:, k], hist_q, hist_qd, hist_tau, lim))
                    tgt_all.append(tau)
                q, qd, _, _ = model.step_contact(q, qd, tau, rcfg.dt, CONTACT)
                last_tau = tau
    obs_all, tgt_all = torch.cat(obs_all), torch.cat(tgt_all)
    opt = torch.optim.Adam(policy.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    n = len(obs_all)
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, 512))
        pred = policy.decode(policy(obs_all[idx]))["torque"]
        loss = ((pred - tgt_all[idx]) / lim).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"[warmstart] step {step} torque mse (normalized) {loss.item():.5f} on {n} samples", flush=True)


def validate(ckpt_path: Path, n_per_regime: int, split: str = "val") -> dict:
    cases = [CaseSpec(r, split, i) for r in REGIMES for i in range(n_per_regime)]
    res = evaluate({"c": {"kind": "learned", "path": str(ckpt_path)}}, cases)["c"]
    per_regime = {r: float(np.mean([m["composite"] for m in res if m["regime"] == r])) for r in REGIMES}
    return {"mean": float(np.mean(list(per_regime.values()))), "per_regime": per_regime,
            "diverged": int(sum(m["diverged"] for m in res))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage-iters", type=int, nargs=3, default=[80, 80, 160])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warmstart", type=int, default=3000, help="supervised steps toward the shared PD torque (neural_torque only)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-cases", type=int, default=2)
    ap.add_argument("--pool", type=int, default=32, help="training references per regime")
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--tbptt", type=int, default=200)
    ap.add_argument("--history", type=int, default=10)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--latency", type=float, default=0.005)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gains", default="runs/m1/gains_shared.json", help="shared PD gains (pd_residual)")
    ap.add_argument("--randomize", action="store_true", help="dynamics randomization during training (Milestone 3, ADR 0004)")
    ap.add_argument("--dynamics", default="known", help="'known' or path to a learned dynamics model checkpoint (Milestone 4)")
    ap.add_argument("--pulses", type=float, default=0.0, help="fraction of training windows with a 10 N disturbance pulse (protocol v2: 0.3)")
    ap.add_argument("--warmstart-gains", default="initial", help="PD gains imitated by the neural-torque warm start: 'initial' (hand-set, stable at 100 Hz) or 'shared'")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nominal = NominalModel()
    inertia = home_inertia_diag(nominal)
    shared = json.load(open(args.gains))["spec"] if Path(args.gains).exists() else {"kp": [0.0] * 7, "kd": [0.0] * 7}
    cfg = PolicyConfig(kind=args.kind, history=args.history, rate_hz=args.rate, latency_s=args.latency,
                       inertia_diag=tuple(inertia.tolist()), kp_shared=tuple(shared["kp"]), kd_shared=tuple(shared["kd"]))
    policy = Policy(cfg)
    model = RigidBodyModel(load_model(), dtype=torch.float32)
    if args.dynamics == "known":
        dynamics = RigidBodyDynamics(model, CONTACT)
    else:
        from llc.learning.learned_model import load_learned_dynamics

        dynamics = load_learned_dynamics(args.dynamics)
        for p_ in dynamics.net.parameters():
            p_.requires_grad_(False)
    rcfg = RolloutConfig(dt=0.0025, horizon_s=args.horizon, tbptt_steps=args.tbptt, contact=CONTACT)
    lcfg = LossConfig()
    pool = reference_pool(args.pool, rcfg.dt, "train")
    randomizer = ModelRandomizer(model, RandomizationRanges()) if args.randomize else None
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    total_iters = sum(args.stage_iters)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_iters, eta_min=args.lr * 0.05)
    log_f = open(out / "log.jsonl", "a")
    meta = {"args": vars(args), "policy_config": cfg.__dict__, "rollout": {"dt": rcfg.dt, "horizon_s": rcfg.horizon_s, "tbptt": rcfg.tbptt_steps},
            "contact": CONTACT.__dict__, "loss": lcfg.__dict__, "stages": STAGES, "git": git_hash(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "randomize": RandomizationRanges().__dict__ if args.randomize else None}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    best = {"val": float("inf"), "iter": -1}
    it, t0 = 0, time.time()
    if args.kind == "neural_torque" and args.warmstart > 0:
        if args.warmstart_gains == "initial":
            from experiments.m1_tune_gains import X0_PD, pd_spec

            ws_gains = pd_spec(X0_PD, inertia)
        else:
            ws_gains = shared
        warm_start(policy, pool, rcfg, args.warmstart, rng, inertia, ws_gains)
        log_f.write(json.dumps({"warmstart_steps": args.warmstart}) + "\n")
    for stage, n_iters in enumerate(args.stage_iters):
        for _ in range(n_iters):
            sampled = sample_batch(pool, STAGES[stage], args.batch, rcfg.horizon_steps, rcfg.dt, rng, pulse_prob=args.pulses)
            q_ref, qd_ref, q0, qd0, labels = sampled[:5]
            tt = lambda x: torch.as_tensor(x, dtype=torch.float32)
            f_ext = tt(sampled[5]) if len(sampled) > 5 else None
            opt.zero_grad()
            dr = randomizer.sample(rng) if randomizer is not None else None
            res = rollout(policy, dynamics, tt(q_ref), tt(qd_ref), tt(q0), tt(qd0), rcfg, lcfg, train=True,
                          latency_s=dr["latency_s"] if dr else None, obs_noise_q=dr["obs_noise_q"] if dr else 0.0, f_ext=f_ext)
            gn = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0))
            opt.step()
            sched.step()
            rec = {"iter": it, "stage": stage, "loss": res["loss"], "terms": res["loss_terms"], "grad_norm": gn, "dr": dr,
                   "max_f": float(torch.linalg.norm(res["f"], dim=-1).max()), "elapsed_s": time.time() - t0}
            if (it + 1) % args.val_every == 0 or it + 1 == total_iters:
                ckpt = out / "last.pt"
                save_policy(ckpt, policy, {"iter": it, "stage": stage})
                val = validate(ckpt, args.val_cases)
                rec["val"] = val
                if val["mean"] < best["val"]:
                    best = {"val": val["mean"], "iter": it}
                    save_policy(out / "best.pt", policy, {"iter": it, "stage": stage, "val": val})
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            msg = f"[{args.kind} s{args.seed}] it {it:4d} st {stage} loss {res['loss']:.3f} " + " ".join(f"{k}={v:.3f}" for k, v in res["loss_terms"].items()) + f" gn {gn:.2f} maxF {rec['max_f']:.0f} ({rec['elapsed_s']:.0f}s)"
            if "val" in rec:
                msg += f" | val {rec['val']['mean']:.3f} " + " ".join(f"{k[:4]}={v:.2f}" for k, v in rec["val"]["per_regime"].items())
            print(msg, flush=True)
            it += 1
    save_policy(out / "final.pt", policy, {"iter": it - 1, "stage": 2})
    (out / "best.json").write_text(json.dumps(best))
    print(f"done; best val {best['val']:.3f} at iter {best['iter']}; wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
