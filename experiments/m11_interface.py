"""Milestone 11 (ADR 0011, decision 4): train the learned torque-command interface of a profile through the
plant. The upper layer is the feedforward PD (plus a torque program on hold windows, the torque-tracking
family); the interface's corrections are optimised by backpropagation through the MuJoCo batch
(finite-difference gradients, median clipping) on the loss between the realised torque and the upper
command evaluated at the true state at every physics step, plus the torque-rate penalty.

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/m11_interface.py --profile p2 --gains runs/m9/gains_shared.json --out runs/m11/iface_p2-s0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from llc.controllers.program import PROGRAMS, program_torque
from llc.dynamics.rigid_body import RigidBodyModel
from llc.eval.parallel import CaseSpec, evaluate
from llc.learning.actor_critic import reference_accel
from llc.learning.data import reference_pool_v2, sample_batch_v2
from llc.learning.interface import LIMITS_T, InterfaceConfig, InterfacePolicy, frame_law_torch, interface_obs, model_interface_torch, profile_of, save_interface, st_quantise
from llc.learning.mjgrad import MjBatch
from llc.learning.rollout_mj import _clip_hook
from llc.sim.instances import sample_instance
from llc.sim.nominal import NominalModel
from llc.sim.panda import NUM_JOINTS, load_model

MIX = {"hold": 0.4, "free": 0.2, "policy": 0.1, "contact2": 0.15, "blocked": 0.15}


def _median_clip_hook(kappa: float):
    def hook(g):
        if g is None:
            return None
        n = g.norm(dim=-1, keepdim=True)
        return g * torch.clamp(kappa * n.median() / (n + 1e-12), max=1.0)
    return hook


def sample_programs(labels: list[str], rng: np.random.Generator) -> list[dict | None]:
    """A torque program for hold windows (the torque-tracking family); none elsewhere."""
    out = []
    for lab in labels:
        if lab != "hold":
            out.append(None); continue
        kind = PROGRAMS[int(rng.integers(len(PROGRAMS)))]
        amp = {"hold": 0.0, "step": float(rng.uniform(0.1, 0.3)), "chirp": 0.1, "torque_chirp": 0.02}[kind]
        out.append({"kind": kind, "joint": int(rng.integers(7)), "amplitude": amp, "t_start": float(rng.uniform(0.1, 0.4)), "f0": 0.5, "f1": 125.0, "duration": float(rng.uniform(0.6, 1.5))})
    return out


def rollout_interface(policy: InterfacePolicy, batch: MjBatch, model: RigidBodyModel, q_ref, qd_ref, q0, qd0, programs, f_ext=None, train: bool = False,
                      tbptt: int = 100, dt: float = 1e-3, rate_weight: float = 0.003, rate_scale: float = 500.0, median_clip: float = 3.0, obs_noise_q: float = 1e-4,
                      state_grad: str = "clipped") -> dict:
    """The servo loop through the batch: frames at the host rate from the feedforward PD (plus programs), the
    model-based interface and the policy's corrections, the bus delay, the servo law at the physics rate.
    Returns the loss terms (realised against commanded torque, torque rate) and detached logs."""
    p = profile_of(policy.cfg)
    cfg = policy.cfg
    B, T, _ = q_ref.shape
    kp_n, kd_n = policy.kp_n.expand(B, NUM_JOINTS), policy.kd_n.expand(B, NUM_JOINTS)
    batch.reset(np.asarray(q0, np.float64), np.asarray(qd0, np.float64))
    q, qd = (x.float() for x in batch.state())
    qdd_ref = reference_accel(qd_ref, dt)
    H = cfg.history
    hist = {k: q[:, None].repeat(1, H, 1).clone() for k in ("q", "goal")}
    hist["qd"] = torch.zeros(B, H, NUM_JOINTS); hist["tau"] = torch.zeros(B, H, NUM_JOINTS)
    last_tau, last_goal = torch.zeros(B, NUM_JOINTS), q.clone()
    pending: list = []
    held = held_upper = latest = None
    k_frame = 0
    period, delay = p.period_steps, p.delay_steps
    hook = _median_clip_hook(median_clip) if median_clip > 0 else _clip_hook(1e3)
    terms = {"track": 0.0, "rate": 0.0}
    logs = {"tau": [], "tau_cmd": [], "q": []}
    cap = LIMITS_T.expand(B, NUM_JOINTS)
    for k0 in range(0, T, tbptt):
        k1 = min(T, k0 + tbptt)
        ct = {k: torch.zeros(()) for k in terms}
        for k in range(k0, k1):
            q_o = q + obs_noise_q * torch.randn_like(q) if obs_noise_q > 0 else q
            qd_o = qd + 10 * obs_noise_q * torch.randn_like(qd) if obs_noise_q > 0 else qd
            q_m, qd_m = st_quantise(q_o, p.pos_res), st_quantise(qd_o, p.vel_res)
            if k % period == 0:
                base = model.inverse_dynamics(q_m, qd_ref[:, k], qdd_ref[:, k])
                prog = torch.as_tensor(np.stack([program_torque(pr, k * dt) if pr is not None else np.zeros(NUM_JOINTS) for pr in programs]), dtype=torch.float32)
                pure = torch.tensor([pr is not None and pr["kind"] == "torque_chirp" for pr in programs])[:, None]
                law = kp_n * (q_ref[:, k] - q_m) + kd_n * (qd_ref[:, k] - qd_m) + base + prog
                upper = {"q_des": q_ref[:, k], "qd_des": qd_ref[:, k], "kp": torch.where(pure, torch.zeros_like(kp_n), kp_n), "kd": torch.where(pure, torch.zeros_like(kd_n), kd_n),
                         "tau_ff": torch.where(pure, law, base + prog), "cap": cap}
                servo = model_interface_torch(p, policy.kp_n, policy.kd_n, upper, q_m, qd_m)
                for key, val in (("q", q_m), ("qd", qd_m), ("tau", last_tau), ("goal", last_goal)):
                    hist[key] = torch.cat([hist[key][:, 1:], val.detach()[:, None]], 1)
                obs = interface_obs(cfg, {kk: vv.detach() for kk, vv in upper.items()}, q_m.detach(), qd_m.detach(), hist["q"], hist["qd"], hist["tau"], hist["goal"], policy.kp_n, policy.kd_n)
                servo = policy.apply(p, servo, policy.decode(policy(obs)))
                pending.append((k + delay, servo, upper))
                latest, last_goal, k_frame = upper, servo["q_des"], k
            while pending and k >= pending[0][0]:
                _, held, held_upper = pending.pop(0)
            if held is None:
                g = model.bias(q_m, torch.zeros_like(q_m))
                held = {"q_des": q_m.detach(), "qd_des": torch.zeros_like(q_m), "kp": kp_n, "kd": kd_n, "tau_ff": g.detach(), "cap": cap}
                held_upper = held
            tau = frame_law_torch(held, q_m, qd_m)
            lat = latest if latest is not None else held_upper
            prog_now = torch.as_tensor(np.stack([program_torque(pr, k * dt) - program_torque(pr, k_frame * dt) if pr is not None else np.zeros(NUM_JOINTS) for pr in programs]), dtype=torch.float32)
            tau_cmd = (frame_law_torch({**lat, "q_des": q_ref[:, k], "qd_des": qd_ref[:, k]}, q, qd) + prog_now).detach()  # the 1 kHz command: a target, not a path
            ct["track"] = ct["track"] + ((tau - tau_cmd) / LIMITS_T).square().sum(-1).mean()
            ct["rate"] = ct["rate"] + rate_weight * ((tau - last_tau) / (dt * rate_scale)).square().sum(-1).mean()
            q, qd, touch, _ = batch.step(q, qd, tau, None if f_ext is None else f_ext[:, k])
            q, qd = q.float(), qd.float()
            if state_grad == "none":  # the correction's gradient through the servo law only (the plant's response within a hold is second order)
                q, qd = q.detach(), qd.detach()
            elif train and q.requires_grad:
                q.register_hook(hook); qd.register_hook(hook)
            last_tau = tau
            logs["tau"].append(tau.detach()); logs["tau_cmd"].append(tau_cmd); logs["q"].append(q.detach())
        loss = sum(ct.values()) / T
        if train:
            loss.backward()
        for name in terms:
            terms[name] += ct[name].item() / T
        q, qd, last_tau, last_goal = q.detach(), qd.detach(), last_tau.detach(), last_goal.detach()
        pending = [(kk, {a: b.detach() for a, b in s.items()}, {a: b.detach() for a, b in u.items()}) for kk, s, u in pending]
        held = None if held is None else {a: b.detach() for a, b in held.items()}
        held_upper = None if held_upper is None else {a: b.detach() for a, b in held_upper.items()}
        latest = None if latest is None else {a: b.detach() for a, b in latest.items()}
    out = {k: torch.stack(v, 1) for k, v in logs.items()}
    out["terms"] = terms
    out["loss"] = sum(terms.values())
    return out


def validate(path, profile: str, overrides: dict, gains: dict, n_torque: int, n_free: int, workers: int | None) -> dict:
    """Validation through the servo loop: the torque-tracking family (the torque program) and the hold, free and
    blocked families under the feedforward PD; the mean cap composite per family and over families."""
    iface = {"kind": "learned", "path": str(path)}
    wrap = lambda inner: {"kind": "servo", "profile": profile, "overrides": overrides, "inner": inner, "interface": iface, "kp": gains["kp"], "kd": gains["kd"]}
    pd = {**gains, "kind": "pd_feedforward"}
    specs = {"prog": wrap({"kind": "torque_program", "base": pd}), "pd": wrap(pd)}
    cases_t = [CaseSpec("torque", "val", i, protocol="v2") for i in range(n_torque)]
    cases_f = [CaseSpec(r, "val", i, protocol="v2") for r in ("hold", "free", "blocked") for i in range(n_free)]
    res_t = evaluate({"prog": specs["prog"]}, cases_t, workers=workers)["prog"]
    res_f = evaluate({"pd": specs["pd"]}, cases_f, workers=workers)["pd"]
    per = {"torque": float(np.mean([m["composite_cap"] for m in res_t]))}
    for r in ("hold", "free", "blocked"):
        per[r] = float(np.mean([m["composite_cap"] for m in res_f if m["regime"] == r]))
    return {"mean": float(np.mean(list(per.values()))), "per_regime": per, "tau_track_rmse": float(np.mean([m["tau_track_rmse"] for m in res_t])),
            "tau_track_delay": float(np.nanmean([m["tau_track_delay"] for m in res_t])), "torque_rate": float(np.mean([m["torque_rate_rms"] for m in res_t + res_f])),
            "diverged": int(sum(m["diverged"] for m in res_t + res_f))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--period", type=float, default=None)
    ap.add_argument("--delay", type=float, default=None)
    ap.add_argument("--gains", default="runs/m9/gains_shared.json", help="the servo's nominal gains and the upper PD's")
    ap.add_argument("--history", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--goal-bound", type=float, default=0.05)
    ap.add_argument("--tau-bound", type=float, default=0.2)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--tbptt", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-rate", type=float, default=0.003)
    ap.add_argument("--median-clip", type=float, default=3.0)
    ap.add_argument("--state-grad", default="clipped", choices=["clipped", "none"], help="none: detach the plant state, the gradient flows through the servo law only")
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--val-torque", type=int, default=8)
    ap.add_argument("--val-free", type=int, default=2)
    ap.add_argument("--pool", type=int, default=48)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--sim-threads", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gains = json.load(open(args.gains))["spec"]
    overrides = {k: v for k, v in (("period", args.period), ("delay", args.delay)) if v is not None}
    cfg = InterfaceConfig(args.profile, tuple(gains["kp"]), tuple(gains["kd"]), args.history, args.hidden, args.layers, args.goal_bound, (0.5, 2.0), args.tau_bound, overrides or None)
    policy = InterfacePolicy(cfg)
    nominal = NominalModel()
    model = RigidBodyModel(load_model(), dtype=torch.float32)
    pool = reference_pool_v2(args.pool, nominal=nominal)
    horizon = int(round(args.horizon / 1e-3))
    batch = MjBatch([sample_instance(rng, payload_mass=float(rng.uniform(0, 3))) for _ in range(args.batch)], seed=args.seed, threads=args.sim_threads)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    meta = {"args": vars(args), "method_key": f"iface_{args.variant or args.profile}", "config": cfg.__dict__, "git": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log_f = open(out / "log.jsonl", "a")
    save_interface(out / "last.pt", policy, {"iter": -1})
    val = validate(out / "last.pt", args.profile, overrides, gains, args.val_torque, args.val_free, args.workers)
    best = {"val": val["mean"], "iter": -1}
    save_interface(out / "best.pt", policy, {"iter": -1, "val": val})
    log_f.write(json.dumps({"iter": -1, "val": val}) + "\n"); log_f.flush()
    print(f"[iface {args.profile} s{args.seed}] warm start (model-based): val {val['mean']:.3f} track {val['tau_track_rmse']:.4f} delay {1e3*val['tau_track_delay']:.0f}ms rate {val['torque_rate']:.0f} " + " ".join(f"{k[:4]}={v:.2f}" for k, v in val["per_regime"].items()), flush=True)
    t0 = time.time()
    for it in range(args.iters):
        q_ref, qd_ref, q0, qd0, labels, f_ext, _ = sample_batch_v2(pool, MIX, args.batch, horizon, 1e-3, rng, pulse_prob=0.3, start_mode="mixed")
        batch.perturb([sample_instance(rng, payload_mass=float(rng.uniform(0, 3))) for _ in range(args.batch)])
        programs = sample_programs(labels, rng)
        policy.train()
        opt.zero_grad()
        res = rollout_interface(policy, batch, model, torch.as_tensor(q_ref), torch.as_tensor(qd_ref), q0, qd0, programs, f_ext, train=True, tbptt=args.tbptt, rate_weight=args.w_rate, median_clip=args.median_clip, state_grad=args.state_grad)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        policy.eval()
        rec = {"iter": it, "loss": res["loss"], **{f"loss_{k}": v for k, v in res["terms"].items()}, "elapsed_s": time.time() - t0}
        if (it + 1) % args.val_every == 0 or it == args.iters - 1:
            save_interface(out / "last.pt", policy, {"iter": it})
            save_interface(out / f"ckpt_{it:04d}.pt", policy, {"iter": it})
            val = validate(out / "last.pt", args.profile, overrides, gains, args.val_torque, args.val_free, args.workers)
            rec["val"] = val
            if val["mean"] < best["val"]:
                best = {"val": val["mean"], "iter": it}
                save_interface(out / "best.pt", policy, {"iter": it, "val": val})
            print(f"[iface {args.profile} s{args.seed}] iter {it} loss {res['loss']:.5f} (track {res['terms']['track']:.5f} rate {res['terms']['rate']:.5f}) | val {val['mean']:.3f} track {val['tau_track_rmse']:.4f} delay {1e3*val['tau_track_delay']:.0f}ms rate {val['torque_rate']:.0f} " + " ".join(f"{k[:4]}={v:.2f}" for k, v in val["per_regime"].items()) + f" ({time.time()-t0:.0f}s)", flush=True)
        elif it % 5 == 0:
            print(f"[iface {args.profile} s{args.seed}] iter {it} loss {res['loss']:.5f} (track {res['terms']['track']:.5f} rate {res['terms']['rate']:.5f}) ({time.time()-t0:.0f}s)", flush=True)
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
    (out / "best.json").write_text(json.dumps(best))
    print(f"done; best val {best['val']:.3f} at iter {best['iter']}; wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
