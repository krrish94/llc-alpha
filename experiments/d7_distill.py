"""Deep dive D6 (ADR 0008): distil the privileged contact-impedance oracle into a proprioceptive
controller by DAgger, warm-started from the D3 residual.

    uv run python experiments/d7_distill.py --warmstart runs/d4/pd_residual_acmed_s02_lowrate-s0/best.pt \
        --gain-residual --ext-torque --out runs/d7/distill_gain_ext-s0

Each iteration rolls out one episode per sampled training case (regime by the contact-focused
mixture; the teacher drives with probability beta, else the student), labels every student decision
with the teacher's torque, aggregates, fits the student by torque regression and validates on the
32 fixed cases under the cap. The teacher is the oracle around the frozen warm start (the student's
own free-space behaviour, so that imitation only changes what happens in contact) or around the
shared PD.
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

from experiments.d5_ppo import FAMILIES
from llc.controllers.oracle import ORACLE_DEFAULTS
from llc.eval.parallel import CaseSpec, validate_policy
from llc.learning.distill import LabelStore, collect_labels, fit_student
from llc.learning.policy import Policy, PolicyConfig, convert_to_gain_residual, expand_ext_torque, expand_inputs, expand_preview, load_policy, save_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", default="", help="policy checkpoint to start from; empty: a fresh policy (host-rate students of ADR 0011)")
    ap.add_argument("--profile", default="", help="interface profile (ADR 0011): episodes run through the profile's servo loop, the student decides at every host frame")
    ap.add_argument("--period", type=float, default=None)
    ap.add_argument("--delay", type=float, default=None)
    ap.add_argument("--vel-filter", type=float, default=None, help="s; the profile's velocity-feedback filter (ADR 0011 amendment 3)")
    ap.add_argument("--interface", default="model", help="model, or the path of a learned interface checkpoint")
    ap.add_argument("--layers", type=int, default=2, help="fresh policy: trunk depth")
    ap.add_argument("--hidden", type=int, default=128, help="fresh policy: trunk width")
    ap.add_argument("--teacher-base", default="warmstart", choices=["warmstart", "pd"])
    ap.add_argument("--free-teacher", default="oracle", choices=["oracle", "ideal"], help="ideal: the free-space families are labelled with the TAM-style ideal-robot torque instead of the oracle (two teachers, ADR 0010 amendment 2 item D)")
    ap.add_argument("--gains", default="runs/d1/gains_shared.json")
    ap.add_argument("--gain-residual", action="store_true")
    ap.add_argument("--ext-torque", action="store_true")
    ap.add_argument("--preview", type=int, default=0, help="number of future targets (10 ms apart) in the observation; variant")
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--families", default="contact_focus", choices=sorted(FAMILIES))
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--cases", type=int, default=48, help="episodes per iteration")
    ap.add_argument("--betas", default="1,0.5,0.25", help="probability that the teacher drives, per iteration (last value repeats)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-rows", type=int, default=300000)
    ap.add_argument("--free-weight", type=float, default=1.0, help="label-loss weight of decisions from the free-space families (hold, free, policy, step); the free-space fix of ADR 0010")
    ap.add_argument("--contact-weight", type=float, default=1.0, help="label-loss weight of decisions from the contact families (contact-weighted labels, ADR 0011 amendment 3)")
    ap.add_argument("--force-weight", type=float, default=0.0, help="weight of the force-space label term: the force along the surface normal the torque error implies, per the force cap, at decisions in contact (ADR 0011 amendment 3)")
    ap.add_argument("--dist", default="v3", choices=["v3", "v5"], help="training distribution of the collection episodes (ADR 0012: v5 = the consolidation distribution with the base gains and the latency drawn per episode)")
    ap.add_argument("--val-families", default="", help="comma-separated families validated (default: the seven of protocol v2; 'all': every family of protocol v5)")
    ap.add_argument("--expand-inputs", action="store_true", help="add the deployment inputs (base gain scales, compute latency) to the warm start with zero columns (ADR 0012)")
    ap.add_argument("--residual-blocks", action="store_true", help="fresh policy: residual trunk with layer normalisation")
    ap.add_argument("--val-cases", type=int, default=2)
    ap.add_argument("--val-cases-contact", type=int, default=8)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="")
    ap.add_argument("--out", required=True)
    for k, v in ORACLE_DEFAULTS.items():
        ap.add_argument(f"--oracle-{k}", type=float, default=v)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    oracle = {k: getattr(args, f"oracle_{k}") for k in ORACLE_DEFAULTS}

    pd_spec = json.load(open(args.gains))["spec"]
    servo = None
    if args.profile:
        from llc.sim.servo import get_profile

        overrides = {k: v for k, v in (("period", args.period), ("delay", args.delay), ("vel_filter", args.vel_filter)) if v is not None}
        prof = get_profile(args.profile, **overrides)
        servo = {"profile": args.profile, "overrides": overrides, "kp": pd_spec["kp"], "kd": pd_spec["kd"], "period": prof.period, "interface": "model" if args.interface == "model" else {"kind": "learned", "path": args.interface}}
    if args.warmstart:
        src, src_extra = load_policy(args.warmstart)
        if args.gain_residual and src.cfg.kind == "pd_residual":
            src = convert_to_gain_residual(src)
        if args.ext_torque and not src.cfg.ext_torque:
            src = expand_ext_torque(src)
        if args.preview > 0:
            src = expand_preview(src, args.preview, 10)
        if args.expand_inputs and src.cfg.extra_dim == 0:
            src = expand_inputs(src)
        policy = Policy(replace(src.cfg, hold_steps=args.hold))
        policy.load_state_dict(src.state_dict())
    else:  # a fresh student at the host rate (ADR 0011): neutral outputs reproduce the feedforward PD through the interface
        from llc.controllers.pd import home_inertia_diag
        from llc.sim.nominal import NominalModel

        assert servo is not None, "a fresh student needs a profile (its history runs at the host rate)"
        hist = int(np.clip(round(0.1 / servo["period"]), 5, 25))  # 100 ms of history at the host rate
        cfg = PolicyConfig(kind="pd_gain_residual" if args.gain_residual else "pd_residual", history=hist, history_stride=1, history_dense=0, rate_hz=1.0 / servo["period"], latency_s=0.0,
                           hidden=args.hidden, layers=args.layers, inertia_diag=tuple(home_inertia_diag(NominalModel())), kp_shared=tuple(pd_spec["kp"]), kd_shared=tuple(pd_spec["kd"]),
                           ext_torque=args.ext_torque, feedforward=True, residual_blocks=args.residual_blocks, base_scale_input=args.expand_inputs, latency_input=args.expand_inputs)
        policy, src_extra = Policy(cfg), {}
        assert args.teacher_base == "pd", "a fresh student's teacher is the oracle around the PD"
    base = {"kind": "learned", "path": str(args.warmstart)} if args.teacher_base == "warmstart" else {**pd_spec, "kind": "pd_feedforward"} if not args.warmstart else pd_spec
    teacher = {"kind": "oracle", "base": base, "params": oracle, "free_teacher": args.free_teacher, **({"servo": servo} if servo else {})}
    weights = FAMILIES[args.families]
    regimes = list(weights); p = np.array([weights[r] for r in regimes]); p /= p.sum()
    betas = [float(b) for b in args.betas.split(",")]
    val_families = None
    if args.val_families == "all":
        from llc.eval.cases_v2 import REGIMES_V2, TASK_REGIMES

        val_families = list(REGIMES_V2) + list(TASK_REGIMES)
    elif args.val_families:
        val_families = args.val_families.split(",")
    validate = lambda path: validate_policy(path, args.val_cases, args.val_cases_contact, "composite_cap", args.workers, servo=servo, families=val_families)
    meta = {"args": vars(args), "method_key": f"distill_{args.variant or args.families}", "policy_config": policy.cfg.__dict__, "teacher": teacher, "families": weights,
            "warmstart_extra": src_extra, "git": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(), "protocol": "v2", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log_f = open(out / "log.jsonl", "a")
    store = LabelStore(args.max_rows, rng)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    t0 = time.time()
    save_policy(out / "last.pt", policy, {"iter": -1})
    val = validate(out / "last.pt")
    best = {"val": val["mean"], "iter": -1}
    save_policy(out / "best.pt", policy, {"iter": -1, "val": val})
    log_f.write(json.dumps({"iter": -1, "val": val}) + "\n"); log_f.flush()
    print(f"[distill {args.variant} s{args.seed}] warm start: val {val['mean']:.3f} exc {val['excess_force_cap']:.1f} " + " ".join(f"{k[:4]}={v:.2f}" for k, v in val["per_regime"].items()), flush=True)
    for it in range(args.iters):
        beta = betas[min(it, len(betas) - 1)]
        cases = [CaseSpec(regimes[rng.choice(len(regimes), p=p)], "train", int(rng.integers(1000, 100000)), protocol="v2", dist=args.dist) for _ in range(args.cases)]
        execute_teacher = [bool(rng.uniform() < beta) for _ in cases]
        t_c = time.time()
        batches = collect_labels(out / "last.pt", teacher, cases, execute_teacher, args.workers)
        n_new = int(sum(len(b["obs"]) for b in batches)); n_div = int(sum(int(b["diverged"][0]) for b in batches))
        store.add(batches, weight_of=lambda r: args.free_weight if r in ("hold", "free", "policy", "step") else args.contact_weight)
        t_f = time.time()
        fit = fit_student(policy, opt, store, args.epochs, args.batch, rng, force_weight=args.force_weight)
        save_policy(out / "last.pt", policy, {"iter": it})
        save_policy(out / f"ckpt_{it:03d}.pt", policy, {"iter": it})
        val = validate(out / "last.pt")
        rec = {"iter": it, "beta": beta, "n_new": n_new, "n_diverged": n_div, "n_rows": len(store), **fit, "collect_s": t_f - t_c, "fit_s": time.time() - t_f, "elapsed_s": time.time() - t0, "val": val,
               "regimes": {r: sum(c.regime == r for c in cases) for r in regimes}}
        if val["mean"] < best["val"]:
            best = {"val": val["mean"], "iter": it}
            save_policy(out / "best.pt", policy, {"iter": it, "val": val})
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        print(f"[distill {args.variant} s{args.seed}] iter {it} beta {beta:.2f} rows {len(store)} (+{n_new}, div {n_div}) loss {fit['loss_first']:.4f} -> {fit['loss_last']:.4f} "
              f"| val {val['mean']:.3f} exc {val['excess_force_cap']:.1f} press {val['press_force']:.1f} loss/s {val['contact_loss_rate']:.1f} " + " ".join(f"{k[:4]}={v:.2f}" for k, v in val["per_regime"].items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    save_policy(out / "final.pt", policy, {"iter": args.iters - 1})
    (out / "best.json").write_text(json.dumps(best))
    print(f"done; best val {best['val']:.3f} at iter {best['iter']}; wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
