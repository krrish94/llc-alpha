"""Blocked-family diagnosis of a host-rate student (ADR 0011 amendment 3, item 2): where does the student's yield
in blocked contact go missing?

Collects teacher labels along student-driven episodes through the student's profile (the DAgger collection path:
train split, fresh instances and payloads) and reports, per family, the label fit (RMS torque error between the
student's fast-loop torque and the teacher's, in Nm, and the normalised loss the regression minimises) next to the
teacher's and the student's yield from the feedforward PD base (RMS over joints of tau - tau_base); for the
oracle-labelled families also per reference phase (approach, blocked, release; approach, press, slide, release),
with the student's mean gain multipliers and residual magnitude. Checkpoints are compared: the iteration-0 one
(fitted on the teacher's own episodes only) against the selected one.

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/m11_blocked_diag.py --run runs/m11/student_p1-s0 --ckpts ckpt_000.pt best.pt --out runs/m11/blocked_diag_p1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from llc.eval.parallel import CaseSpec, _get_case, _init_worker, _worker
from llc.learning.distill import collect_labels
from llc.learning.policy import load_policy

FAMILIES = ("blocked", "contact2", "hold", "free")
PHASED = ("blocked", "contact2")


def _stats(policy, b: dict, rows: np.ndarray | None = None, normal=None, nominal=None) -> dict:
    """Fit and yield statistics over the rows of one batch (all rows, or a boolean selection). With the surface normal
    (world frame, pointing out of the surface) and the nominal model, the torque error is projected on the normal: the
    extra force the student's torque pushes into the surface compared with the teacher's, by the least-squares force
    that explains the torque difference through the tool Jacobian (positive: the student presses harder)."""
    sel = slice(None) if rows is None else rows
    t = {k: torch.as_tensor(b[k][sel]) for k in ("obs", "q", "qd", "q_ref", "qd_ref", "gravity", "tau_teacher")}
    if len(t["obs"]) == 0:
        return {}
    with torch.no_grad():
        held = policy.decode(policy(t["obs"]))
        tau_s = policy.fast_torque(held, t["q"], t["qd"], t["q_ref"], t["qd_ref"], t["gravity"])
    base = policy.kp_shared * (t["q_ref"] - t["q"]) + policy.kd_shared * (t["qd_ref"] - t["qd"]) + t["gravity"]
    err = tau_s - t["tau_teacher"]
    out = {"n": int(len(t["obs"])), "fit_rms_nm": float(err.square().mean().sqrt()), "fit_loss": float((err / policy.limits).square().mean()),
           "yield_teacher_rms_nm": float((t["tau_teacher"] - base).square().mean().sqrt()), "yield_student_rms_nm": float((tau_s - base).square().mean().sqrt()),
           "yield_corr": float(torch.corrcoef(torch.stack([(t["tau_teacher"] - base).flatten(), (tau_s - base).flatten()]))[0, 1]) if len(t["obs"]) > 2 else float("nan")}
    if "kp_scale" in held:
        out["kp_scale_mean"] = float(held["kp_scale"].mean()); out["kd_scale_mean"] = float(held["kd_scale"].mean())
        out["residual_rms_nm"] = float(held["residual"].square().mean().sqrt())
    if normal is not None and nominal is not None:
        n = np.asarray(normal, float)
        q = t["q"].numpy(); e = err.numpy()
        df = []
        for i in range(0, len(q), max(1, len(q) // 400)):  # a subsample of the rows
            jp = nominal.tool_jacobian(q[i])[0]  # (3, n_joints) position Jacobian of the tool point
            jn = jp.T @ n  # the joint torque of a unit force along the normal
            df.append(-float(jn @ e[i]) / max(1e-9, float(jn @ jn)))  # the student's extra push into the surface (the force on the tool is along -n)
        out["push_error_mean_n"] = float(np.mean(df)); out["push_error_rms_n"] = float(np.sqrt(np.mean(np.square(df))))
    return out


def _accumulate(parts: list[dict]) -> dict:
    """Row-weighted means of per-batch statistics."""
    parts = [p for p in parts if p]
    if not parts:
        return {}
    n = np.array([p["n"] for p in parts], float)
    out = {"n": int(n.sum())}
    for k in parts[0]:
        if k == "n":
            continue
        v = np.array([p[k] for p in parts], float)
        out[k] = float(np.sqrt(np.nansum(n * v * v) / n.sum())) if k.endswith("_nm") else float(np.nansum(n * v) / n.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpts", nargs="*", default=["ckpt_000.pt", "best.pt"])
    ap.add_argument("--cases", type=int, default=8, help="episodes per family (train split, fixed seeds)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    meta = json.load(open(run / "meta.json"))
    teacher = meta["teacher"]
    period = teacher["servo"]["period"] if teacher.get("servo") else 0.01
    cases = [CaseSpec(r, "train", 20000 + 17 * i, protocol="v2") for r in FAMILIES for i in range(args.cases)]
    if not _worker:
        _init_worker()
    nominal = _worker["nominal"]
    phases = {c.regime + str(c.index): _get_case(c).reference.phases for c in cases if c.regime in PHASED}
    normals = {c.regime + str(c.index): _get_case(c).meta.get("normal") for c in cases if c.regime in PHASED}
    result = {"run": str(run), "period": period, "families": {}, "checkpoints": {}}
    for ckpt in args.ckpts:
        policy, extra = load_policy(run / ckpt)
        batches = collect_labels(str(run / ckpt), teacher, cases, [False] * len(cases), args.workers)
        rec = {"iter": extra.get("iter"), "per_family": {}, "per_phase": {}}
        for r in FAMILIES:
            bs = [b for b in batches if str(b["regime"][0]) == r]
            rec["per_family"][r] = _accumulate([_stats(policy, b) for b in bs])
            rec["per_family"][r]["diverged"] = int(sum(int(b["diverged"][0]) for b in bs))
            if r in PHASED:
                labels = sorted({lab for c in cases if c.regime == r for _, _, lab in phases[r + str(c.index)]}, key=lambda s: ("approach", "press", "slide", "blocked", "release", "free").index(s) if s in ("approach", "press", "slide", "blocked", "release", "free") else 9)
                for lab in labels:
                    parts = []
                    for c, b in zip([c for c in cases if c.regime == r], bs):
                        t = np.arange(len(b["obs"])) * period  # the oracle path logs one row per host frame
                        mask = np.zeros(len(t), bool)
                        for t0, t1, l in phases[r + str(c.index)]:
                            if l == lab:
                                mask |= (t >= t0) & (t < t1)
                        parts.append(_stats(policy, b, mask, normal=normals[r + str(c.index)] if lab in ("blocked", "press", "slide") else None, nominal=nominal))
                    rec["per_phase"][f"{r}/{lab}"] = _accumulate(parts)
        rec["val"] = extra.get("val", {}).get("mean") if isinstance(extra.get("val"), dict) else None
        result["checkpoints"][ckpt] = rec
        print(f"[{run.name} {ckpt} iter {rec['iter']}]")
        for key, s in list(rec["per_family"].items()) + list(rec["per_phase"].items()):
            if not s or "fit_rms_nm" not in s:
                continue
            extra_s = f" kp x{s['kp_scale_mean']:.2f} kd x{s['kd_scale_mean']:.2f} |res| {s['residual_rms_nm']:.2f} Nm" if "kp_scale_mean" in s else ""
            extra_s += f"  push error {s['push_error_mean_n']:+.1f} N (rms {s['push_error_rms_n']:.1f})" if "push_error_mean_n" in s else ""
            print(f"   {key:18s} n {s['n']:6d}  fit {s['fit_rms_nm']:.2f} Nm (loss {s['fit_loss']:.4f})  yield teacher {s['yield_teacher_rms_nm']:.2f} student {s['yield_student_rms_nm']:.2f} Nm (corr {s['yield_corr']:.2f}){extra_s}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
