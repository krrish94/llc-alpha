"""Mode-detection probe (deep dive): does a learned controller's hidden representation carry the
contact state? Roll the policy on protocol v2 validation cases, record the last hidden layer and
the measured contact flag at every step, fit a linear probe by logistic regression on half of the
episodes, and report accuracy, balanced accuracy and the detection lead/lag around contact onset
on the other half. A probe on the raw observation serves as the reference: if the raw inputs
already separate the modes, the network need not have learned anything.

    uv run python experiments/d2_probe.py --run runs/d2/pd_residual-s0 --out runs/d2/probe_pd_residual-s0.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from llc.controllers.learned import LearnedController
from llc.eval.cases_v2 import make_case_v2
from llc.learning.policy import load_policy
from llc.sim.nominal import NominalModel
from llc.sim.panda import PandaSim


def collect(policy, nominal, cases):
    """Per step: hidden activation (last trunk layer), raw observation, contact flag, time to onset."""
    feats, raws, flags, episodes = [], [], [], []
    captured = {}

    def hook(_m, _i, out):
        captured["h"] = out.detach()[0].numpy()

    handle = policy.trunk[-1].register_forward_hook(hook)
    orig_forward = policy.forward

    def forward(obs):
        captured["obs"] = obs.detach()[0].numpy()
        return orig_forward(obs)

    policy.forward = forward
    for ep_i, case in enumerate(cases):
        sim = PandaSim(case.perturbation, seed=case.seed)
        ctrl = LearnedController(policy, nominal)
        ref = case.reference
        sim.reset(case.q0, case.qd0)
        for k in range(len(ref.t)):
            q, qd = sim.observe()
            tau = ctrl(ref.t[k], q, qd, ref.q[k], ref.qd[k], ref.qdd[k])
            sim.apply_tool_force(np.zeros(3))
            sim.step(tau)
            if k % 5 == 0:
                feats.append(captured["h"].copy()); raws.append(captured["obs"].copy()); flags.append(sim.contact().n_tool_contacts > 0); episodes.append(ep_i)
    handle.remove()
    policy.forward = orig_forward
    return np.array(feats), np.array(raws), np.array(flags), np.array(episodes)


def logistic_probe(x_tr, y_tr, x_te, y_te, steps=600, lr=0.1):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    xt, xe = torch.as_tensor((x_tr - mu) / sd, dtype=torch.float32), torch.as_tensor((x_te - mu) / sd, dtype=torch.float32)
    yt = torch.as_tensor(y_tr, dtype=torch.float32)
    w = torch.zeros(xt.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    pos_w = float((1 - yt.mean()) / max(yt.mean(), 1e-6))
    for _ in range(steps):
        logits = xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, pos_weight=torch.tensor(pos_w)) + 1e-3 * (w * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(xe @ w + b).numpy()
    pred = p > 0.5
    acc = float((pred == y_te).mean())
    tpr = float((pred[y_te]).mean()) if y_te.any() else float("nan")
    tnr = float((~pred[~y_te]).mean()) if (~y_te).any() else float("nan")
    return {"accuracy": acc, "balanced_accuracy": float(np.nanmean([tpr, tnr])), "tpr": tpr, "tnr": tnr}, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-cases", type=int, default=6)
    args = ap.parse_args()
    policy, _ = load_policy(Path(args.run) / "best.pt")
    nominal = NominalModel()
    cases = [make_case_v2(r, "val", i, nominal) for r in ("contact2", "blocked", "free", "task") for i in range(args.n_cases)]
    feats, raws, flags, episodes = collect(policy, nominal, cases)
    train = episodes % 2 == 0
    res_hidden, p_h = logistic_probe(feats[train], flags[train], feats[~train], flags[~train])
    res_raw, p_r = logistic_probe(raws[train], flags[train], raws[~train], flags[~train])
    # detection timing around contact onsets in test episodes: first probe crossing vs first measured contact
    leads_h, leads_r = [], []
    te_eps = np.unique(episodes[~train])
    for e in te_eps:
        idx = np.where((episodes == e) & ~train)[0]
        f = flags[idx]
        if not f.any():
            continue
        k_on = int(np.argmax(f))
        for p, store in ((p_h, leads_h), (p_r, leads_r)):
            pp = p[np.searchsorted(np.where(~train)[0], idx)]
            cross = np.where(pp > 0.5)[0]
            store.append(float((cross[0] - k_on) * 5) if cross.size else float("nan"))
    out = {"run": args.run, "n_samples_test": int((~train).sum()), "contact_fraction": float(flags.mean()), "hidden_probe": res_hidden, "raw_probe": res_raw,
           "onset_lead_ms_hidden": {"median": float(np.nanmedian(leads_h)) if leads_h else None, "values": leads_h},
           "onset_lead_ms_raw": {"median": float(np.nanmedian(leads_r)) if leads_r else None, "values": leads_r}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("onset_lead_ms_hidden", "onset_lead_ms_raw")}, indent=1))
    print("onset lead (ms, negative = before measured contact): hidden", out["onset_lead_ms_hidden"]["median"], "raw", out["onset_lead_ms_raw"]["median"])


if __name__ == "__main__":
    main()
