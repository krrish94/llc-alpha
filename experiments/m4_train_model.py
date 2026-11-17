"""Milestone 4: train and validate a learned dynamics model on plant data.

    uv run python experiments/m4_train_model.py --variant blackbox --data runs/m4/data_train.npz --val runs/m4/data_val.npz --out runs/m4/model_blackbox

Validation reports three separate claims: (1) prediction error at 1 step, 100 ms and 1 s horizons
per regime (open loop, applying the recorded torques); (2) gradient usefulness: agreement of the
sensitivity d qd(t+100 ms) / d tau(t) with the known rigid-body model in free space (cosine
similarity and relative norm); (3) closed-loop control is assessed separately by training
controllers through the model (experiments/m2_train.py --dynamics ...).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from llc.dynamics.rigid_body import RigidBodyModel
from llc.learning.learned_model import LearnedDynamics, LearnedDynamicsNet, LearnedModelConfig, save_model
from llc.sim.panda import load_model


def transitions(episodes, dt_model: float, dt_data: float = 1e-3):
    """(q, qd, tau_mean, qd_next, f_next) at the model step from 1 kHz episodes; tau is averaged over the step."""
    stride = int(round(dt_model / dt_data))
    rows = {k: [] for k in ("q", "qd", "tau", "qd_next", "f", "regime")}
    for ep in episodes:
        n = (len(ep["q"]) - 1) // stride * stride
        q, qd, tau, f = ep["q"][:n + 1], ep["qd"][:n + 1], ep["tau"][:n + 1], ep["f_normal"][:n + 1]
        tau_mean = tau[:n].reshape(-1, stride, 7).mean(1)
        rows["q"].append(q[:n:stride]); rows["qd"].append(qd[:n:stride]); rows["tau"].append(tau_mean)
        rows["qd_next"].append(qd[stride::stride]); rows["f"].append(f[stride::stride]); rows["regime"] += [ep["regime"]] * (n // stride)
    return {k: (np.concatenate(v) if k != "regime" else np.array(v)) for k, v in rows.items()}


def multistep_error(dyn: LearnedDynamics, ep: dict, horizon_steps: int, dt_model: float, n_starts: int = 20) -> float:
    """RMS joint position error after rolling the model open loop for ``horizon_steps`` from several starts."""
    stride = int(round(dt_model / 1e-3))
    q, qd, tau = (torch.as_tensor(ep[k]) for k in ("q", "qd", "tau"))
    n = len(q)
    starts = np.linspace(0, n - horizon_steps * stride - 1, n_starts).astype(int)
    errs = []
    with torch.no_grad():
        qk, qdk = q[starts], qd[starts]
        for s in range(horizon_steps):
            idx = starts + s * stride
            tau_mean = torch.stack([tau[i:i + stride].mean(0) for i in idx])
            qk, qdk, _, _ = dyn.step(qk, qdk, tau_mean, dt_model)
        target = q[starts + horizon_steps * stride]
        errs.append(((qk - target) ** 2).mean().sqrt().item())
    return float(np.mean(errs))


def gradient_agreement(dyn: LearnedDynamics, known: RigidBodyModel, data: dict, dt_model: float, n: int = 64, horizon: int = 40) -> dict:
    """Compare d qd(t+horizon) / d tau(t) between the learned and the known model on free-space data points."""
    free = np.where(data["f"] < 0.1)[0]
    idx = np.random.default_rng(0).choice(free, min(n, len(free)), replace=False)
    cos, rel = [], []
    for i in idx:
        q0 = torch.as_tensor(data["q"][i])[None]; qd0 = torch.as_tensor(data["qd"][i])[None]; tau0 = torch.as_tensor(data["tau"][i])[None]
        jac = {}
        for name, step in (("learned", lambda q, qd, t: dyn.step(q, qd, t, dt_model)[:2]), ("known", lambda q, qd, t: known.step(q, qd, t, dt_model))):
            def f(tau):
                q, qd = q0, qd0
                for _ in range(horizon):
                    q, qd = step(q, qd, tau)  # constant torque perturbation held over the horizon
                return qd[0]
            jac[name] = torch.autograd.functional.jacobian(f, tau0)[:, 0]
        a, b = jac["learned"].flatten(), jac["known"].flatten()
        cos.append(float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)))
        rel.append(float((a - b).norm() / (b.norm() + 1e-12)))
    return {"n": int(len(idx)), "cosine_mean": float(np.mean(cos)), "cosine_min": float(np.min(cos)), "rel_norm_error_mean": float(np.mean(rel))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="blackbox")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--dt", type=float, default=0.0025)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_eps = list(np.load(args.data, allow_pickle=True)["episodes"])
    val_eps = list(np.load(args.val, allow_pickle=True)["episodes"])
    tr, va = transitions(train_eps, args.dt), transitions(val_eps, args.dt)
    qdd_tr = (tr["qd_next"] - tr["qd"]) / args.dt
    known = RigidBodyModel(load_model(), dtype=torch.float32)
    cfg = LearnedModelConfig(variant=args.variant, dt=args.dt, q_mean=tuple(tr["q"].mean(0).tolist()), q_std=tuple((tr["q"].std(0) + 1e-3).tolist()),
                             qd_std=tuple((tr["qd"].std(0) + 1e-3).tolist()), qdd_std=tuple((qdd_tr.std(0) + 1e-3).tolist()))
    net = LearnedDynamicsNet(cfg)
    dyn = LearnedDynamics(net, known)
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    q, qd, tau, qd_next, f = (T(tr[k]) for k in ("q", "qd", "tau", "qd_next", "f"))
    qv, qdv, tauv, qd_nextv, fv = (T(va[k]) for k in ("q", "qd", "tau", "qd_next", "f"))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=1e-5)
    rng = np.random.default_rng(0)
    log = []
    t0 = time.time()
    qdd_scale = torch.as_tensor(cfg.qdd_std)
    for step in range(args.steps):
        idx = torch.as_tensor(rng.integers(0, len(q), args.batch))
        qdd_pred, f_pred = dyn.acceleration(q[idx], qd[idx], tau[idx])
        qdd_true = (qd_next[idx] - qd[idx]) / args.dt
        loss = ((qdd_pred - qdd_true) / qdd_scale).square().mean() + 0.1 * ((f_pred - f[idx]) / cfg.f_scale).square().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 1000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                qdd_v, f_v = dyn.acceleration(qv, qdv, tauv)
                vloss = ((qdd_v - (qd_nextv - qdv) / args.dt) / qdd_scale).square().mean().item()
                verr = ((qdd_v - (qd_nextv - qdv) / args.dt)).square().mean().sqrt().item()
            log.append({"step": step, "train_loss": loss.item(), "val_loss": vloss, "val_qdd_rmse": verr, "elapsed_s": time.time() - t0})
            print(f"[{args.variant}] step {step} train {loss.item():.4f} val {vloss:.4f} (qdd rmse {verr:.2f} rad/s^2) {time.time()-t0:.0f}s", flush=True)
    save_model(out / "model.pt", net, {"train_episodes": len(train_eps), "steps": args.steps})
    # validation claims
    net.eval()
    horizons = {"1 step (2.5 ms)": 1, "100 ms": int(round(0.1 / args.dt)), "1 s": int(round(1.0 / args.dt))}
    per_regime = {}
    for ep in val_eps:
        r = ep["regime"]
        per_regime.setdefault(r, {h: [] for h in horizons})
        for h, steps in horizons.items():
            if len(ep["q"]) > steps * int(round(args.dt / 1e-3)) + 2:
                per_regime[r][h].append(multistep_error(dyn, ep, steps, args.dt))
    pred = {r: {h: float(np.mean(v)) if v else float("nan") for h, v in hs.items()} for r, hs in per_regime.items()}
    grad = gradient_agreement(dyn, known, va, args.dt)
    summary = {"variant": args.variant, "config": cfg.__dict__, "log": log, "prediction_rmse_rad": pred, "gradient_agreement": grad,
               "n_train_transitions": int(len(q)), "n_val_transitions": int(len(qv))}
    json.dump(summary, open(out / "validation.json", "w"), indent=1, default=str)
    print(json.dumps({"prediction_rmse_rad": pred, "gradient_agreement": grad}, indent=1))


if __name__ == "__main__":
    main()
