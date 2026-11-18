"""Probing study (Milestone 11, ADR 0011 decision 7c): what do a learned controller's trunk layers encode?

Replays a learned controller on validation episodes of the surface families, the free-space families
and the compliant-surface and payload groups, records the features of every trunk layer at every
decision, and fits linear probes (fitted on even episodes, tested on odd ones) to privileged labels the
simulator knows: contact (yes/no), surface identity (table or wall, among contact decisions), the surface
normal in the tool frame (three components, among contact decisions), the surface stiffness class (rigid,
2 kN/m, 500 N/m, among contact decisions), the payload mass and the external force magnitude on the tool.
Reports balanced accuracy (classification) or R² (regression) per layer, and for contact the detection
delay after an onset. A probe from the raw observation is the reference ("is it already in the input?").

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/m11_probe.py --out runs/m11/probe.json student=runs/m10r2/distill_two-s0/best.pt
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

FAMILIES = ("contact2", "blocked", "hold", "free")


def replay(policy, nominal: NominalModel, ep, q_ref, qd_ref, profile=None) -> dict:
    """The controller's decisions along a recorded episode; the features of every trunk layer per decision. Through a servo
    profile (ADR 0011) the policy decides once per host frame on the quantised feedback, as the servo loop calls it."""
    from llc.sim.servo import quantise

    period = 1 if profile is None else profile.period_steps
    ctrl = LearnedController(policy, nominal) if profile is None else LearnedController(policy, nominal, dt=profile.period, latency_s=0.0)
    feats: dict[int, list] = {}
    captured = {}
    hooks = []
    for i, layer in enumerate(policy.trunk):
        if isinstance(layer, torch.nn.ELU):
            hooks.append(layer.register_forward_hook(lambda m, inp, out, i=i: captured.__setitem__(i, out.detach()[0].numpy())))
    idx, obs = [], []
    n = len(ep.t)
    for k in range(n):
        before = ctrl.n_updates
        if profile is None:
            ctrl(ep.t[k], ep.q[k], ep.qd[k], q_ref[k], qd_ref[k], None)
        elif k % period == 0:
            ctrl.frame(ep.t[k], quantise(ep.q[k], profile.pos_res), quantise(ep.qd[k], profile.vel_res), q_ref[k], qd_ref[k], np.zeros_like(q_ref[k]))
        ctrl.last_tau = ep.tau[k].copy()
        if ctrl.n_updates > before:
            for i, f in captured.items():
                feats.setdefault(i, []).append(f.copy())
            idx.append(k); obs.append(ctrl.last_obs[0].numpy().copy())
    for h in hooks:
        h.remove()
    return {"layers": {i: np.array(v) for i, v in feats.items()}, "obs": np.array(obs), "steps": np.array(idx)}


def fit_linear(X, y, l2=1e-2):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    w = np.linalg.solve(Xn.T @ Xn + l2 * len(X) * np.eye(Xn.shape[1]), Xn.T @ y)
    return lambda F: np.concatenate([(F - mu) / sd, np.ones((len(F), 1))], 1) @ w


def fit_logistic(X, y, steps=300):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float32); yt = torch.as_tensor(y, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    pos = yt.mean().clamp(1e-3, 1 - 1e-3)
    weight = torch.where(yt > 0.5, 0.5 / pos, 0.5 / (1 - pos))
    for _ in range(steps):
        loss = (weight * torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt, reduction="none")).mean() + 1e-3 * (w * w).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    w, b = w.detach(), b.detach()
    return lambda F: (torch.as_tensor((F - mu) / sd, dtype=torch.float32) @ w + b).sigmoid().numpy()


def fit_mlp(X, y, classify: bool, hidden: int = 64, steps: int = 400):
    """A nonlinear probe (one hidden layer of 64) next to the linear ones (ADR 0011 amendment 3, item 6)."""
    import torch.nn.functional as F

    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float32); yt = torch.as_tensor(y, dtype=torch.float32)
    yt = yt[:, None] if yt.ndim == 1 else yt
    y_mu, y_sd = (torch.zeros(yt.shape[1]), torch.ones(yt.shape[1])) if classify else (yt.mean(0), yt.std(0) + 1e-6)  # regression targets standardised
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ELU(), torch.nn.Linear(hidden, yt.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
    if classify:
        pos = yt.mean().clamp(1e-3, 1 - 1e-3)
        w = torch.where(yt > 0.5, 0.5 / pos, 0.5 / (1 - pos))
    for _ in range(steps):
        out = net(Xt)
        loss = (w * F.binary_cross_entropy_with_logits(out, yt, reduction="none")).mean() if classify else F.mse_loss(out, (yt - y_mu) / y_sd)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def pred(Fx):
        with torch.no_grad():
            o = net(torch.as_tensor((Fx - mu) / sd, dtype=torch.float32))
        o = o.sigmoid() if classify else o * y_sd + y_mu
        return o.numpy()[:, 0] if o.shape[1] == 1 else o.numpy()
    return pred


def balanced_accuracy(pred, y):
    y = y.astype(bool)
    if not y.any() or (~y).any() is False:
        return float("nan")
    return float(0.5 * (pred[y].mean() + (~pred[~y]).mean())) if y.any() and (~y).any() else float("nan")


def r2(pred, y):
    return float(1.0 - np.sum((pred - y) ** 2) / max(1e-12, np.sum((y - y.mean(0)) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="name=path pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=int, default=8, help="validation cases per family and group")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--profile", default="", help="interface profile: the controllers run and are replayed through its servo loop (host-rate students)")
    ap.add_argument("--gains", default="runs/m9/gains_shared.json", help="the servo's nominal gains through a profile")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    nominal = NominalModel()
    profile = None
    if args.profile:
        from llc.sim.servo import get_profile

        profile = get_profile(args.profile)
        gains = json.load(open(args.gains))["spec"]
        wrap = lambda spec: {"kind": "servo", "profile": args.profile, "inner": spec, "interface": "model", "kp": gains["kp"], "kd": gains["kd"]}
    else:
        wrap = lambda spec: spec
    specs = [CaseSpec(r, "val", i, protocol="v2") for r in FAMILIES for i in range(args.cases)]
    specs += [CaseSpec(r, "val", i, protocol="v2", surface_stiffness=k) for k in (2000.0, 500.0) for r in ("contact2", "blocked") for i in range(args.cases // 2)]
    specs += [CaseSpec(r, "val", i, protocol="v2", payload=3.0) for r in ("free", "contact2") for i in range(args.cases // 2)]
    from llc.sim.plant import PLANT

    if PLANT.tasks:  # the pick with a grasp: is the block in the gap before the grasp instant? (amendment 3, item 6)
        specs += [CaseSpec("pick", "val", i, protocol="v2") for i in range(args.cases // 2)]
    out = {}
    for item in args.checkpoints:
        name, path = item.split("=")
        policy, _ = load_policy(path)
        res = evaluate({"c": wrap({"kind": "learned", "path": path})}, specs, keep_episodes=True, workers=args.workers)["c"]
        per_ep = []
        for spec, m in zip(specs, res):
            case = make_case_v2(spec.regime, spec.split, spec.index, nominal, payload=spec.payload, surface_stiffness=spec.surface_stiffness)
            ep = m["episode"]; ref = case.reference; n = len(ep.t)
            rp = replay(policy, nominal, ep, ref.q[:n], ref.qd[:n], profile)
            st = rp["steps"]
            contact = ep.n_tool_contacts[st] > 0
            surface = case.meta.get("surface")
            normal_w = np.asarray(case.meta.get("normal") or [0.0, 0.0, 1.0], float)
            # the surface normal in the tool frame at every decision
            normal_t = np.array([nominal.tool_frame(ep.q[k])[1].T @ normal_w for k in st])
            # the external force magnitude on the tool: the contact force plus the hold pulse
            f_ext = ep.f_normal[st].copy()
            if case.pulse is not None:
                on = (ep.t[st] >= case.pulse.t_start) & (ep.t[st] < case.pulse.t_start + case.pulse.duration)
                f_ext[on] += np.linalg.norm(case.pulse.force)
            stiff_class = {None: 0, 2000.0: 1, 500.0: 2}[spec.surface_stiffness]
            # extended targets (amendment 3, item 6): the reference's depth inside the surface, sliding against pressing, the
            # phase of a blocked episode, and for the pick whether the block is in the gap before the grasp
            ref_path = ref.meta.get("tool_path")  # the reference's tool path (the case's meta drops it)
            depth = surface_depth(np.asarray(ref_path, float)[:n][st], surface) if (ref_path is not None and surface in ("table", "wall")) else np.full(len(st), np.nan)
            v = np.gradient(ep.tool_pos, ep.t, axis=0)
            sliding = np.linalg.norm((v - np.outer(v @ normal_w, normal_w))[st], axis=1) > 0.02
            blocked_phase = ref.phase_mask("blocked")[:n][st] if spec.regime == "blocked" else None
            release_phase = ref.phase_mask("release")[:n][st] if spec.regime == "blocked" else None
            in_gap = pre_grasp = None
            task = case.meta.get("task") or {}
            if spec.regime == "pick" and ep.obj_pos is not None:
                d = ep.obj_pos[st] - ep.tool_pos[st]
                tol = task.get("grasp_tol", (0.006, 0.009))
                in_gap = (np.linalg.norm(d[:, :2], axis=1) <= tol[0]) & (np.abs(d[:, 2]) <= tol[1])
                pre_grasp = ep.t[st] < float(task.get("grasp_time", np.inf))
            per_ep.append({"regime": spec.regime, "layers": rp["layers"], "obs": rp["obs"], "contact": contact, "wall": np.full(len(st), surface == "wall"),
                           "normal": normal_t, "stiff": np.full(len(st), stiff_class), "payload": np.full(len(st), float(case.perturbation.payload_mass)), "f_ext": f_ext, "f_normal": ep.f_normal[st],
                           "steps": st, "dt": float(np.median(np.diff(ep.t))), "has_surface": surface is not None, "depth": depth, "sliding": sliding, "blocked_phase": blocked_phase, "release_phase": release_phase,
                           "in_gap": in_gap, "pre_grasp": pre_grasp})
        train = [e for i, e in enumerate(per_ep) if i % 2 == 0]; test = [e for i, e in enumerate(per_ep) if i % 2 == 1]
        layer_ids = sorted(per_ep[0]["layers"])
        sources = {"input": lambda e: e["obs"], **{f"layer{j + 1}": (lambda e, i=i: e["layers"][i]) for j, i in enumerate(layer_ids)}}
        rec = {"n_decisions": int(sum(len(e["steps"]) for e in per_ep)), "contact_fraction": float(np.concatenate([e["contact"] for e in per_ep]).mean())}
        for src_name, src in sources.items():
            r = {}
            # contact: all decisions
            X = np.concatenate([src(e) for e in train]); y = np.concatenate([e["contact"] for e in train]).astype(float)
            pred = fit_logistic(X, y)
            yt = np.concatenate([e["contact"] for e in test]); pt = np.concatenate([pred(src(e)) for e in test]) > 0.5
            r["contact_balanced_accuracy"] = balanced_accuracy(pt, yt)
            delays, n_on = [], 0
            for e in test:
                p = pred(src(e)) > 0.5; c = e["contact"]
                for o in np.where(c[1:] & ~c[:-1])[0] + 1:
                    end = o + np.argmax(~c[o:]) if (~c[o:]).any() else len(c)
                    n_on += 1
                    after = np.where(p[o:end])[0]
                    if after.size:
                        delays.append(float((e["steps"][o + after[0]] - e["steps"][o]) * e["dt"]))
            r["contact_delay_median_s"] = float(np.median(delays)) if delays else float("nan"); r["contact_within_50ms"] = float(sum(d <= 0.05 for d in delays) / max(1, n_on))
            # among contact decisions: surface identity, normal, stiffness class
            sel = lambda es, key: (np.concatenate([src(e)[e["contact"]] for e in es if e["has_surface"]]), np.concatenate([e[key][e["contact"]] for e in es if e["has_surface"]]))
            Xc, yw = sel(train, "wall"); Xt_, ywt = sel(test, "wall")
            if yw.any() and (~yw).any():
                pw = fit_logistic(Xc, yw.astype(float))
                r["wall_balanced_accuracy"] = balanced_accuracy(pw(Xt_) > 0.5, ywt)
            Xc, yn = sel(train, "normal"); Xt_, ynt = sel(test, "normal")
            r["normal_r2"] = r2(fit_linear(Xc, yn)(Xt_), ynt)
            Xc, ys = sel(train, "stiff"); Xt_, yst = sel(test, "stiff")
            for cls, lab in ((1, "stiff_2000"), (2, "stiff_500")):
                if (ys == cls).any() and (ys != cls).any():
                    ps = fit_logistic(Xc, (ys == cls).astype(float))
                    r[f"{lab}_balanced_accuracy"] = balanced_accuracy(ps(Xt_) > 0.5, yst == cls)
            # payload and external force: all decisions
            X = np.concatenate([src(e) for e in train]); Xt_ = np.concatenate([src(e) for e in test])
            r["payload_r2"] = r2(fit_linear(X, np.concatenate([e["payload"] for e in train]))(Xt_), np.concatenate([e["payload"] for e in test]))
            r["f_ext_r2"] = r2(fit_linear(X, np.concatenate([e["f_ext"] for e in train]))(Xt_), np.concatenate([e["f_ext"] for e in test]))
            r["f_normal_r2"] = r2(fit_linear(X, np.concatenate([e["f_normal"] for e in train]))(Xt_), np.concatenate([e["f_normal"] for e in test]))  # the contact force alone (the pulse's direction is random)
            # extended targets (amendment 3, item 6)
            def subset(es, key, cond):
                parts = [(src(e)[cond(e)], e[key][cond(e)]) for e in es if e.get(key) is not None]
                parts = [(a, b) for a, b in parts if len(a)]
                return (np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])) if parts else (None, None)
            with_surface = lambda e: e["contact"] & e["has_surface"] & np.isfinite(e["depth"])
            Xd, yd = subset(train, "depth", with_surface); Xdt, ydt = subset(test, "depth", with_surface)
            if Xd is not None and Xdt is not None:
                r["depth_r2"] = r2(fit_linear(Xd, yd)(Xdt), ydt)
                r["depth_mlp_r2"] = r2(fit_mlp(Xd, yd, False)(Xdt), ydt)
            for key, cond, lab in (("sliding", lambda e: e["contact"], "sliding"), ("blocked_phase", lambda e: np.ones(len(e["steps"]), bool), "blocked_phase"),
                                   ("release_phase", lambda e: np.ones(len(e["steps"]), bool), "release_phase"), ("in_gap", lambda e: e["pre_grasp"], "in_gap")):
                Xa, ya = subset(train, key, cond); Xb, yb = subset(test, key, cond)
                if Xa is not None and Xb is not None and ya.any() and (~ya).any() and yb.any() and (~yb).any():
                    r[f"{lab}_balanced_accuracy"] = balanced_accuracy(fit_logistic(Xa, ya.astype(float))(Xb) > 0.5, yb)
            # nonlinear probes of the two main targets
            X = np.concatenate([src(e) for e in train]); y = np.concatenate([e["contact"] for e in train]).astype(float)
            r["contact_mlp_balanced_accuracy"] = balanced_accuracy(fit_mlp(X, y, True)(np.concatenate([src(e) for e in test])) > 0.5, np.concatenate([e["contact"] for e in test]))
            Xc, yn = sel(train, "normal"); Xt_, ynt = sel(test, "normal")
            r["normal_mlp_r2"] = r2(fit_mlp(Xc, yn, False)(Xt_), ynt)
            rec[src_name] = r
            print(f"{name:20s} {src_name:8s} " + " ".join(f"{k.replace('_balanced_accuracy', ' acc').replace('_r2', ' R2')}={v:.2f}" for k, v in r.items()), flush=True)
        out[name] = rec
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
