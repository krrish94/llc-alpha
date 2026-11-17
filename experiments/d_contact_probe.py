"""Does the policy sense contact, and does it become compliant? (contact deep dive D5, ADR 0007)

Replays a learned controller on protocol v2 validation cases of the surface families (the policy's
own decisions are reproduced from the recorded trajectory) and records, at every decision, the
trunk features, the decoded outputs (gain multipliers and residual) and the contact state. Reports:

* contact sensing: a logistic probe from the trunk features to "tool in contact" and to "blocked"
  (in contact while the reference is inside the surface), fitted on half of the episodes and tested
  on the other half (accuracy, balanced accuracy), the detection delay after an onset (first decision
  at which the probe exceeds 0.5) and the fraction of onsets detected within 50 and 100 ms (the
  contact-sensing metrics of ADR 0008);
* compliance: the gain multipliers in and out of contact (gain-residual policies), and the residual
  torque mapped to a tool-space force along the surface normal in and out of contact (all policies),
  positive values push away from the surface (along the outward normal).

    uv run python experiments/d_contact_probe.py --out runs/d6/probe.json name=path/to/best.pt ...
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from llc.controllers.learned import LearnedController
from llc.eval.cases_v2 import make_case_v2
from llc.eval.parallel import CaseSpec, evaluate
from llc.learning.data import surface_depth
from llc.learning.policy import load_policy
from llc.sim.nominal import NominalModel


def replay(policy, nominal: NominalModel, ep, q_ref, qd_ref) -> dict:
    """Run the controller's decision logic along a recorded episode; returns per-decision arrays."""
    ctrl = LearnedController(policy, nominal)
    feats, outs, idx = [], [], []
    captured = {}
    hook = policy.trunk.register_forward_hook(lambda m, i, o: captured.__setitem__("f", o.detach()[0].numpy()))
    n = len(ep.t)
    from llc.learning.policy import preview_indices

    for k in range(n):
        before = ctrl.n_updates
        if policy.cfg.preview:
            ctrl(ep.t[k], ep.q[k], ep.qd[k], q_ref[k], qd_ref[k], None, q_ref_future=q_ref[preview_indices(policy.cfg, k, n)])
        else:
            ctrl(ep.t[k], ep.q[k], ep.qd[k], q_ref[k], qd_ref[k], None)
        ctrl.last_tau = ep.tau[k].copy()  # the recorded torque, not the replayed one, keeps the history identical to the episode
        if ctrl.n_updates > before:
            feats.append(captured["f"]); outs.append({kk: v.copy() for kk, v in ctrl.held.items()}); idx.append(k)
    hook.remove()
    return {"features": np.array(feats), "outputs": outs, "steps": np.array(idx)}


def tool_normal_force(nominal: NominalModel, q: np.ndarray, residual: np.ndarray, normal: np.ndarray) -> float:
    """Force along the surface normal that a joint-torque residual produces at the tool (N): f = J^+T tau."""
    J, _ = nominal.tool_jacobian(q)
    f = np.linalg.lstsq(J.T, residual, rcond=None)[0]
    return float(f @ normal)


def fit_probe(X: np.ndarray, y: np.ndarray, steps: int = 400) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float32); yt = torch.as_tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    pos = yt.mean().clamp(1e-3, 1 - 1e-3)
    weight = torch.where(yt > 0.5, 0.5 / pos, 0.5 / (1 - pos))  # class-balanced
    for _ in range(steps):
        loss = (weight * torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt, reduction="none")).mean() + 1e-3 * (w * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach(), b.detach(), mu, sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="name=path pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=int, default=8, help="validation cases per surface family")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    nominal = NominalModel()
    specs = [CaseSpec(r, "val", i, protocol="v2") for r in ("contact2", "blocked") for i in range(args.cases)]
    out = {}
    for item in args.checkpoints:
        name, path = item.split("=")
        policy, _ = load_policy(path)
        res = evaluate({"c": {"kind": "learned", "path": path}}, specs, keep_episodes=True, workers=args.threads)["c"]
        per_ep = []
        for spec, m in zip(specs, res):
            case = make_case_v2(spec.regime, spec.split, spec.index, nominal)
            ep = m["episode"]
            ref = case.reference
            n = len(ep.t)
            rp = replay(policy, nominal, ep, ref.q[:n], ref.qd[:n])
            steps = rp["steps"]
            contact = (ep.n_tool_contacts[steps] > 0)
            inside = surface_depth(np.asarray(case.meta["tool_ref"])[: n], case.meta["surface"])[steps] > 0 if case.meta.get("surface") is not None else np.zeros_like(contact)
            normal = np.asarray(case.meta.get("normal") or [0.0, 0.0, 1.0], float)
            f_res = np.array([tool_normal_force(nominal, ep.q[k], o["residual"], normal) for k, o in zip(steps, rp["outputs"])]) if "residual" in rp["outputs"][0] else None
            kp = np.array([o["kp_scale"] for o in rp["outputs"]]) if "kp_scale" in rp["outputs"][0] else None
            per_ep.append({"regime": spec.regime, "features": rp["features"], "contact": contact, "blocked": contact & inside, "f_res": f_res, "kp": kp, "force": ep.f_normal[steps], "steps": steps, "dt": float(np.median(np.diff(ep.t)))})
        # probes: fit on even episodes, test on odd ones
        train = [e for i, e in enumerate(per_ep) if i % 2 == 0]; test = [e for i, e in enumerate(per_ep) if i % 2 == 1]
        rec = {"contact_fraction": float(np.concatenate([e["contact"] for e in per_ep]).mean())}
        for label in ("contact", "blocked"):
            X = np.concatenate([e["features"] for e in train]); y = np.concatenate([e[label] for e in train]).astype(float)
            w, b, mu, sd = fit_probe(X, y)
            pred = lambda F: (torch.as_tensor((F - mu) / sd, dtype=torch.float32) @ w + b).sigmoid().numpy()
            yt = np.concatenate([e[label] for e in test]); pt = np.concatenate([pred(e["features"]) for e in test]) > 0.5
            acc = float((pt == yt).mean()); bal = float(0.5 * ((pt[yt]).mean() + (~pt[~yt]).mean())) if yt.any() and (~yt).any() else float("nan")
            delays, n_on = [], 0
            for e in test:
                p = pred(e["features"]) > 0.5
                c = e[label]
                for o in np.where(c[1:] & ~c[:-1])[0] + 1:
                    end = o + np.argmax(~c[o:]) if (~c[o:]).any() else len(c)
                    n_on += 1
                    after = np.where(p[o:end])[0]
                    if after.size:
                        delays.append(float((e["steps"][o + after[0]] - e["steps"][o]) * e["dt"]))
            pre = "" if label == "contact" else "blocked_"
            rec.update({f"{pre}probe_accuracy": acc, f"{pre}probe_balanced_accuracy": bal, f"{pre}detection_delay_median_s": float(np.median(delays)) if delays else float("nan"),
                        f"{pre}n_onsets": n_on, f"{pre}within_50ms": float(sum(d <= 0.05 for d in delays) / max(1, n_on)), f"{pre}within_100ms": float(sum(d <= 0.1 for d in delays) / max(1, n_on))})
        acc, bal, delays = rec["probe_accuracy"], rec["probe_balanced_accuracy"], [rec["detection_delay_median_s"]]
        allc = np.concatenate([e["contact"] for e in per_ep])
        if per_ep[0]["kp"] is not None:
            kp = np.concatenate([e["kp"] for e in per_ep])
            rec["kp_scale_in_contact"] = kp[allc].mean(0).tolist(); rec["kp_scale_free"] = kp[~allc].mean(0).tolist()
            rec["kp_scale_in_contact_mean"] = float(kp[allc].mean()); rec["kp_scale_free_mean"] = float(kp[~allc].mean())
        if per_ep[0]["f_res"] is not None:
            fr = np.concatenate([e["f_res"] for e in per_ep])
            rec["residual_normal_force_in_contact_N"] = float(fr[allc].mean()); rec["residual_normal_force_free_N"] = float(fr[~allc].mean())
            rec["residual_normal_force_in_contact_std_N"] = float(fr[allc].std())
        out[name] = rec
        print(f"{name:28s} probe acc {acc:.3f} bal {bal:.3f} delay {rec['detection_delay_median_s']*1e3 if rec['detection_delay_median_s'] == rec['detection_delay_median_s'] else float('nan'):.0f} ms ({rec['n_onsets']} onsets, <=50ms {rec['within_50ms']:.2f}) | blocked acc {rec['blocked_probe_accuracy']:.3f} delay {rec['blocked_detection_delay_median_s']*1e3:.0f} ms | "
              + (f"kp scale contact {rec['kp_scale_in_contact_mean']:.2f} free {rec['kp_scale_free_mean']:.2f} | " if "kp_scale_in_contact_mean" in rec else "")
              + (f"residual normal force contact {rec['residual_normal_force_in_contact_N']:+.1f} N free {rec['residual_normal_force_free_N']:+.1f} N" if "residual_normal_force_in_contact_N" in rec else ""), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
