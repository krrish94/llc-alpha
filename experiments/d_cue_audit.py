"""Cue audit (ADR 0008): which hardware-realisable proprioceptive cues carry the contact information,
how fast, and whether they confuse an unmodelled payload with a contact.

Episodes of fixed controllers (the shared PD and the D3 residual) on training-split cases of the
surface families, plus free-motion and hold cases with payloads (training-split payloads of 0 to
3 kg and the held-out 3 kg; no contact at all, the pulses of the hold family excepted). At every
10 ms decision the candidate cue windows are built exactly as a policy observation would hold them
(the two-scale history: 100 ms at 10 ms plus the last 32 ms at 1 ms), and a class-balanced probe
(logistic, and a one-hidden-layer MLP) is fitted on the even cases of both sets and tested on the
odd ones for two labels: contact (tool in contact) and blocked (in contact while the reference is
inside the surface). Reported per cue set: accuracy and balanced accuracy on the odd surface cases,
median detection delay after a contact onset, the fraction of onsets detected within 50 and
100 ms, and the false-positive rate on the odd payload-only episodes (the payload-against-contact
confusion a proprioceptive controller must resolve).

    uv run python experiments/d_cue_audit.py --out runs/d7/cue_audit.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from llc.eval.cases_v2 import make_case_v2
from llc.eval.parallel import CaseSpec, evaluate
from llc.learning.data import surface_depth
from llc.learning.policy import ERR_SCALE, QD_SCALE, Q_SCALE
from llc.sim.nominal import NominalModel
from llc.sim.panda import TORQUE_LIMITS

STRIDE, N_STRIDED, N_DENSE, HOLD = 10, 10, 32, 10
CUES = {
    "error and velocity": ("cur", "hist_q", "hist_qd"),
    "error, velocity and commanded torque (D3 observation)": ("cur", "hist_q", "hist_qd", "hist_tau"),
    "external torque (link-side torque minus the model)": ("hist_ext",),
    "acceleration (1 ms differences of the velocity)": ("hist_qdd",),
    "target preview (100 ms) with the error": ("cur", "preview"),
    "all of the above": ("cur", "hist_q", "hist_qd", "hist_tau", "hist_ext", "hist_qdd", "preview"),
}


def ext_torque(nominal: NominalModel, q, qd, tau, dt) -> np.ndarray:
    qdd = np.vstack([np.zeros((1, 7)), np.diff(qd, axis=0) / dt])
    ext = np.zeros_like(tau)
    for k in range(1, len(q)):
        ext[k] = tau[k] - (nominal.mass_matrix(q[k]) @ qdd[k] + nominal.bias(q[k], qd[k]))
    return ext


def windows(ep, nominal: NominalModel) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Cue blocks at every decision step (after a 100 ms warm-up), each (N, d)."""
    n = len(ep.t); dt = float(ep.t[1] - ep.t[0])
    length = STRIDE * (N_STRIDED - 1) + 1
    idx_s = np.arange(-STRIDE * (N_STRIDED - 1), 1, STRIDE); idx_d = np.arange(-N_DENSE + 1, 1)
    idx = np.concatenate([idx_s, idx_d])
    ext = ext_torque(nominal, ep.q, ep.qd, ep.tau, dt)
    qdd = np.vstack([np.zeros((1, 7)), np.diff(ep.qd, axis=0) / dt])
    steps = np.arange(max(length, N_DENSE) + 1, n, HOLD)
    blocks = {k: [] for k in ("cur", "hist_q", "hist_qd", "hist_tau", "hist_ext", "hist_qdd", "preview")}
    lim = TORQUE_LIMITS
    for k in steps:
        h = k + idx
        q, qd, qr, qdr = ep.q[k], ep.qd[k], ep.q_ref[k], ep.qd_ref[k]
        blocks["cur"].append(np.concatenate([q / Q_SCALE, qd / QD_SCALE, (qr - q) / ERR_SCALE, qdr / QD_SCALE]))
        blocks["hist_q"].append(((ep.q[h] - q) / ERR_SCALE).ravel())
        blocks["hist_qd"].append((ep.qd[h] / QD_SCALE).ravel())
        blocks["hist_tau"].append((ep.tau[h] / lim).ravel())
        blocks["hist_ext"].append((ext[h] / lim).ravel())
        blocks["hist_qdd"].append((qdd[k + idx_d] / 50.0).ravel())
        fut = np.minimum(k + STRIDE * np.arange(1, 11), n - 1)
        blocks["preview"].append(((ep.q_ref[fut] - qr) / ERR_SCALE).ravel())
    return {k: np.array(v, np.float32) for k, v in blocks.items()}, steps


def fit_probe(X, y, hidden: int = 0, steps: int = 300) -> torch.nn.Module:
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt, yt = torch.as_tensor((X - mu) / sd), torch.as_tensor(y, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ELU(), torch.nn.Linear(hidden, 1)) if hidden else torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(net.parameters(), lr=0.01 if hidden else 0.05, weight_decay=1e-4)
    pos = yt.mean().clamp(1e-3, 1 - 1e-3)
    w = torch.where(yt > 0.5, 0.5 / pos, 0.5 / (1 - pos))
    for _ in range(steps):
        loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(net(Xt).squeeze(-1), yt, reduction="none")).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.mu, net.sd = mu, sd
    return net


def predict(net, X) -> np.ndarray:
    with torch.no_grad():
        return (net(torch.as_tensor((X - net.mu) / net.sd)).squeeze(-1) > 0).numpy()


def onset_stats(pred: np.ndarray, label: np.ndarray, steps: np.ndarray, dt: float) -> tuple[list[float], int]:
    """Detection delays (s) for every onset of the label within an episode; missed onsets count in the total."""
    delays, n = [], 0
    on = np.where(label[1:] & ~label[:-1])[0] + 1
    for o in on:
        end = o + np.argmax(~label[o:]) if (~label[o:]).any() else len(label)
        n += 1
        hit = np.where(pred[o:end])[0]
        if hit.size:
            delays.append(float((steps[o + hit[0]] - steps[o]) * dt))
    return delays, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16, help="cases per surface family")
    ap.add_argument("--gains", default="runs/d1/gains_shared.json")
    ap.add_argument("--residual", default="runs/d4/pd_residual_acmed_s02_lowrate-s0/best.pt")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    nominal = NominalModel()
    specs = {"pd": json.load(open(args.gains))["spec"], "residual": {"kind": "learned", "path": args.residual}}
    cases = [CaseSpec(r, "train", i, protocol="v2") for r in ("contact2", "blocked") for i in range(args.n)] + [CaseSpec("task", "train", i, protocol="v2") for i in range(args.n // 2)]
    payload_cases = ([CaseSpec("free", "test", i, protocol="v2", payload=3.0) for i in range(args.n // 2)] + [CaseSpec("hold", "test", i, protocol="v2", payload=3.0) for i in range(4)]
                     + [CaseSpec("free", "train", i, protocol="v2") for i in range(args.n // 2)] + [CaseSpec("hold", "train", i, protocol="v2") for i in range(4)])
    res = evaluate(specs, cases + payload_cases, keep_episodes=True, workers=args.workers)
    out = {"cases": [c.id for c in cases], "payload_cases": [c.id for c in payload_cases], "controllers": {}}
    for name in specs:
        eps = []
        for spec, m in zip(cases + payload_cases, res[name]):
            case = make_case_v2(spec.regime, spec.split, spec.index, nominal, payload=spec.payload)
            ep = m["episode"]
            blocks, steps = windows(ep, nominal)
            contact = ep.n_tool_contacts[steps] > 0
            if case.meta.get("surface") is not None:
                inside = surface_depth(np.asarray(case.meta["tool_ref"])[: len(ep.t)], case.meta["surface"])[steps] > 0
            else:
                inside = np.zeros_like(contact)
            eps.append({"id": spec.id, "surface": case.meta.get("surface") is not None, "blocks": blocks, "steps": steps, "dt": float(ep.t[1] - ep.t[0]),
                        "contact": contact, "blocked": contact & inside})
        surf = [e for e in eps if e["surface"]]; payload = [e for e in eps if not e["surface"]]
        train = [e for i, e in enumerate(surf) if i % 2 == 0]; test = [e for i, e in enumerate(surf) if i % 2 == 1]
        pl_train = [e for i, e in enumerate(payload) if i % 2 == 0]; pl_test = [e for i, e in enumerate(payload) if i % 2 == 1]
        rec = {}
        for cue, parts in CUES.items():
            X = lambda e: np.concatenate([e["blocks"][p] for p in parts], 1)
            Xtr = np.concatenate([X(e) for e in train + pl_train]); Xte = np.concatenate([X(e) for e in test]); Xpl = np.concatenate([X(e) for e in pl_test])
            rec[cue] = {}
            for label in ("contact", "blocked"):
                ytr = np.concatenate([e[label] for e in train + pl_train]).astype(float); yte = np.concatenate([e[label] for e in test])
                for probe, hidden in (("linear", 0), ("mlp", 64)):
                    net = fit_probe(Xtr, ytr, hidden)
                    p = predict(net, Xte)
                    acc = float((p == yte).mean()); bal = float(0.5 * (p[yte].mean() + (~p[~yte]).mean()))
                    delays, n_on = [], 0
                    for e in test:
                        d, n = onset_stats(predict(net, X(e)), e[label], e["steps"], e["dt"]); delays += d; n_on += n
                    fp = float(predict(net, Xpl).mean())
                    rec[cue][f"{label}/{probe}"] = {"accuracy": acc, "balanced_accuracy": bal, "delay_median_s": float(np.median(delays)) if delays else float("nan"),
                                                     "within_50ms": float(sum(d <= 0.05 for d in delays) / max(1, n_on)), "within_100ms": float(sum(d <= 0.1 for d in delays) / max(1, n_on)),
                                                     "n_onsets": n_on, "payload_false_positive": fp, "n_features": int(Xtr.shape[1])}
                    r = rec[cue][f"{label}/{probe}"]
                    print(f"{name:8s} {cue[:44]:44s} {label:7s} {probe:6s} acc {acc:.3f} bal {bal:.3f} delay {1e3 * r['delay_median_s']:5.0f} ms  <=50ms {r['within_50ms']:.2f} <=100ms {r['within_100ms']:.2f}  payload FP {fp:.3f}", flush=True)
        out["controllers"][name] = rec
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
