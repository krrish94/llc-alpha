"""Milestone 10 (ADR 0010): the TAM-style ideal-robot residual on plant v3.

    LLC_PLANT=v3 uv run python experiments/m10_ideal.py --gains runs/m9/gains_shared.json --out runs/m10/ideal-s0
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
from llc.eval.parallel import CaseSpec, validate_policy
from llc.learning.distill import LabelStore
from llc.learning.ideal import collect_ideal_labels, huber_label_loss
from llc.learning.policy import Policy, PolicyConfig, save_policy
from llc.sim.nominal import NominalModel

FREE = {"hold": 0.25, "free": 0.35, "policy": 0.2, "step": 0.2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", default="runs/m9/gains_shared.json")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--cases", type=int, default=48)
    ap.add_argument("--inject-prob", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--huber", type=float, default=0.02)
    ap.add_argument("--max-rows", type=int, default=300000)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    shared = json.load(open(args.gains))["spec"]
    cfg = PolicyConfig(kind="pd_residual", history=10, history_stride=10, history_dense=32, rate_hz=1000.0, latency_s=0.0, hold_steps=args.hold, ext_torque=True,
                       inertia_diag=tuple(home_inertia_diag(NominalModel()).tolist()), kp_shared=tuple(shared["kp"]), kd_shared=tuple(shared["kd"]))
    policy = Policy(cfg)  # zero residual at initialisation: the plain PD drives the first iteration
    regimes = list(FREE); p = np.array([FREE[r] for r in regimes]); p /= p.sum()
    meta = {"args": vars(args), "method_key": f"ideal_{args.variant or 'residual'}", "policy_config": cfg.__dict__, "families": FREE, "git": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
            "protocol": "v3", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "note": "TAM-style ideal-robot residual, MuJoCo re-implementation without multi-robot pretraining (ADR 0010)"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log_f = open(out / "log.jsonl", "a")
    store = LabelStore(args.max_rows, rng)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    t0 = time.time()
    save_policy(out / "last.pt", policy, {"iter": -1})
    val = validate_policy(out / "last.pt", 2, 8, "composite_cap", args.workers)
    best = {"val": val["mean"], "iter": -1}
    save_policy(out / "best.pt", policy, {"iter": -1, "val": val})
    log_f.write(json.dumps({"iter": -1, "val": val}) + "\n"); log_f.flush()
    print(f"[ideal s{args.seed}] PD: val {val['mean']:.3f} " + " ".join(f"{k[:4]}={v:.2f}" for k, v in val["per_regime"].items()), flush=True)
    for it in range(args.iters):
        cases = [CaseSpec(regimes[rng.choice(len(regimes), p=p)], "train", int(rng.integers(1000, 100000)), protocol="v2") for _ in range(args.cases)]
        inject = [bool(rng.uniform() < args.inject_prob) for _ in cases]
        seeds = [int(rng.integers(0, 2**31 - 1)) for _ in cases]
        t_c = time.time()
        batches = collect_ideal_labels(out / "last.pt", cases, inject, seeds, args.workers)
        store.add(batches)
        n_new = int(sum(len(b["obs"]) for b in batches)); n_div = int(sum(int(b["diverged"][0]) for b in batches))
        data = {k: torch.as_tensor(v) for k, v in store.data.items()}
        n = len(data["obs"])
        policy.train(); first = last = None
        for _ in range(args.epochs):
            perm = torch.as_tensor(rng.permutation(n))
            for i0 in range(0, n, args.batch):
                idx = perm[i0:i0 + args.batch]
                loss = huber_label_loss(policy, {k: v[idx] for k, v in data.items()}, args.huber)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
                first = loss.item() if first is None else first; last = loss.item()
        policy.eval()
        save_policy(out / "last.pt", policy, {"iter": it}); save_policy(out / f"ckpt_{it:03d}.pt", policy, {"iter": it})
        val = validate_policy(out / "last.pt", 2, 8, "composite_cap", args.workers)
        rec = {"iter": it, "n_new": n_new, "n_diverged": n_div, "n_rows": n, "loss_first": first, "loss_last": last, "collect_s": time.time() - t_c, "elapsed_s": time.time() - t0, "val": val}
        if val["mean"] < best["val"]:
            best = {"val": val["mean"], "iter": it}; save_policy(out / "best.pt", policy, {"iter": it, "val": val})
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        print(f"[ideal s{args.seed}] iter {it} rows {n} (+{n_new}, div {n_div}) loss {first:.5f} -> {last:.5f} | val {val['mean']:.3f} " + " ".join(f"{k[:4]}={v:.2f}" for k, v in val["per_regime"].items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    save_policy(out / "final.pt", policy, {"iter": args.iters - 1})
    (out / "best.json").write_text(json.dumps(best))
    print(f"done; best val {best['val']:.3f} at iter {best['iter']}")


if __name__ == "__main__":
    main()
