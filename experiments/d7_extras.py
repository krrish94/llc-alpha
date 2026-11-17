"""D6 (impedance by imitation) report extras: the stop-and-wait table with the oracle rows, blocked
traces, contact stiffness against surface stiffness, the cue audit and contact-sensing tables, and
the training curves of the distilled and fine-tuned arms. HTML snippets used by d7_narrative."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, design, method_color, reference_color  # noqa: E402
from report_api import figure, table  # noqa: E402

from experiments.d_report import pooled  # noqa: E402
from llc.learning.data import surface_depth  # noqa: E402

REPORT = Path("research/reports/13-dd-impedance")
FIG = REPORT / "figures"
LABELS = {
    "pd_shared": "PD, shared gains", "pd_feedforward": "PD + inverse-dynamics feedforward", "oracle_pd": "Oracle around the PD (privileged)",
    "oracle_pd_residual_acmed_s02_lowrate-s0": "Oracle around the D3 residual (privileged, the teacher)",
    "pd_residual_acmed_s02_lowrate": "PD + residual, actor-critic (D3 warm start)", "ppo_contact_v2": "PPO, contact families (D4)",
    "ppo_cap_gain": "PPO under the cap, gain residual (D5 stage 1)", "ppo_cap_gain_ext": "PPO under the cap, external-torque input (D5)",
    "distill_gain_ext": "Distilled student (gain residual, external-torque input)", "distill_gain_ext_preview": "Distilled student with a 100 ms target preview",
    "ppo_distill_gain_ext": "Distilled student + PPO under the cap (stop-and-wait reward)", "ppo_cap_gain_ext_anchor": "PPO, external-torque input, stop-and-wait reward only (control)",
}
STIFFNESS_GROUPS = {"in_distribution": 60000.0, "surface_2000": 2000.0, "surface_500": 500.0}


def pooled_key(name: str) -> str:
    """Pooled method key of an episode or run name (the seed suffix dropped)."""
    if name in LABELS or name in design()["methods"]:
        return name
    base = name.rsplit("-s", 1)[0]
    return base if (base in LABELS or base in design()["methods"]) else name


def label(key: str) -> str:
    return LABELS.get(pooled_key(key), key)


def color(key: str) -> str:
    try:
        return method_color(pooled_key(key))
    except KeyError:
        return "#9e9e9e"


def load(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text())
    groups = {g: pooled(r, data["learned"])[0] for g, r in data["groups"].items()}
    return data, groups


def mean_metric(res: list[dict], regime: str, key: str) -> float:
    v = [m[key] for m in res if m["regime"] == regime and key in m]
    return float(np.nanmean(v)) if v else float("nan")


STOP_WAIT_KEYS_V2 = [("excess_force_cap", "Force above 15 N (N)", 1), ("f_peak_20ms", "Peak, 20 ms (N)", 0), ("blocked_drift", "Drift (cm)", 2),
                     ("blocked_motion", "Moving", 2), ("apparent_stiffness", "Apparent K (N/m)", 0), ("contact_impedance", "Contact K (N/m)", 0),
                     ("contact_recovery", "Recovery (s)", 2), ("torque_rate_rms", "Torque rate (Nm/s)", 0)]
STOP_WAIT_KEYS_V3 = STOP_WAIT_KEYS_V2[:6] + [("release_reaction", "Release reaction (s)", 2), ("release_settle", "Release settling (s)", 2), ("torque_rate_rms", "Torque rate (Nm/s)", 0)]


def stop_and_wait_table(groups: dict, methods: list[str], caption: str, keys=None) -> str:
    keys = keys or STOP_WAIT_KEYS_V2
    rows = []
    for mk in methods:
        if mk not in groups["in_distribution"]:
            continue
        res = groups["in_distribution"][mk]
        row = [label(mk)]
        for k, _, d in keys:
            v = mean_metric(res, "blocked", k)
            row.append(f"{(100 * v if k == 'blocked_drift' else v):.{d}f}")
        rows.append(row)
    return table(["Controller"] + [h for _, h, _ in keys], rows, caption=caption, numeric_cols=tuple(range(1, len(keys) + 1)))


def fig_stiffness(groups: dict, methods: list[str], fig_id: str = "fig-stiffness") -> str:
    """Contact stiffness (force per metre of actual error) and mean force against the surface stiffness."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    xs = list(STIFFNESS_GROUPS.values())
    for mk in methods:
        if any(mk not in groups.get(g, {}) for g in STIFFNESS_GROUPS):
            continue
        for ax, (regime, key, lab) in zip(axes, (("blocked", "contact_impedance", "blocked: contact stiffness (N/m)"), ("contact2", "contact_impedance", "press and slide: contact stiffness (N/m)"), ("blocked", "excess_force_cap", "blocked: force above 15 N (N)"))):
            ys = [mean_metric(groups[g][mk], regime, key) for g in STIFFNESS_GROUPS]
            ax.plot(xs, ys, marker="o", ms=4, lw=1.0, color=color(mk), label=label(mk) if ax is axes[0] else None)
            ax.set_ylabel(lab)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("surface stiffness at the tool (N/m)")
        ax.set_xticks(xs); ax.set_xticklabels(["60 k (rigid)", "2 k", "500"])
    axes[0].axhline(500.0, color=reference_color(), lw=0.8, ls=(0, (4, 2)), label="oracle spring, 500 N/m")
    axes[1].axhline(500.0, color=reference_color(), lw=0.8, ls=(0, (4, 2)))
    for ax in axes[:2]:
        ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "stiffness.svg", bbox_inches="tight"); plt.close(fig)
    return figure("figures/stiffness.svg", "Contact stiffness (mean normal force per metre of the tool's actual error along the normal while the reference is inside; 16 blocked and 16 press-and-slide test cases on the rigid plant, 8 each on the compliant surfaces) and the force above the cap, against the surface stiffness. A spring on the error keeps its stiffness on every surface until the force bound; a controller that pushes to a force setpoint does not.", fig_id, wide=True)


def _load_ep(root: Path, case: str, method: str):
    p = root / f"{case}__{method}.npz"
    return np.load(p) if p.exists() else None


def fig_blocked_traces(root: Path, case: str, methods: list[str], normal, surface: str, tool_ref: np.ndarray, fig_id: str = "fig-blocked-traces") -> str:
    """Normal force, tool error along the normal and tangential drift from the first-contact point on one blocked case."""
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True)
    n = np.asarray(normal, float)
    depth_ref = surface_depth(tool_ref, surface)
    for mk in methods:
        ep = _load_ep(root, case, mk)
        if ep is None:
            continue
        t, f, tool = ep["t"], ep["f_normal"], ep["tool_pos"]
        col = color(mk)
        axes[0].plot(t, f, color=col, lw=0.9, label=label(mk))
        e_in = (tool - tool_ref[: len(t)]) @ n
        axes[1].plot(t, 100 * e_in, color=col, lw=0.9)
        inside = depth_ref[: len(t)] > 0
        on = np.where(inside & (ep["n_tool_contacts"] > 0))[0]
        if on.size:
            d = tool - tool[on[0]]
            d_t = d - np.outer(d @ n, n)
            drift = np.linalg.norm(d_t, axis=1); drift[: on[0]] = np.nan; drift[np.where(inside)[0][-1] + 1:] = np.nan  # defined while the reference is inside
            axes[2].plot(t, 100 * drift, color=col, lw=0.9)
    axes[0].axhline(15.0, color=reference_color(), lw=0.8, ls=(0, (4, 2)), label="15 N cap")
    inside = depth_ref > 0
    if inside.any():
        t_ref = np.arange(len(depth_ref)) * 1e-3
        for ax in axes:
            ax.axvspan(t_ref[inside][0], t_ref[inside][-1], color="#000000", alpha=0.05, lw=0)
    axes[0].set_ylabel("normal force (N)"); axes[1].set_ylabel("tool behind the reference\nalong the normal (cm)"); axes[2].set_ylabel("tangential drift from\nfirst contact (cm)")
    axes[2].set_xlabel("time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "blocked_traces.svg", bbox_inches="tight"); plt.close(fig)
    return figure("figures/blocked_traces.svg", f"Blocked target, test case {case}: normal force, the tool's error behind the reference along the normal (the oracle's spring acts on this), and the tangential drift from the first-contact point. Shaded: the reference inside the surface.", fig_id)


def cue_table(path: Path, controller: str = "residual", caption: str | None = None) -> str:
    if not path.exists():
        return ""
    d = json.loads(path.read_text())["controllers"][controller]
    rows = []
    for cue, rec in d.items():
        c_l, c_m, b_m = rec["contact/linear"], rec["contact/mlp"], rec["blocked/mlp"]
        rows.append([cue, str(c_l["n_features"]), f"{c_l['balanced_accuracy']:.2f}", f"{c_m['balanced_accuracy']:.2f}", f"{1e3 * c_m['delay_median_s']:.0f}", f"{c_m['within_50ms']:.2f}", f"{c_m['within_100ms']:.2f}",
                     f"{c_m['payload_false_positive']:.2f}", f"{b_m['balanced_accuracy']:.2f}", f"{1e3 * b_m['delay_median_s']:.0f}"])
    return table(["Cue set", "Inputs", "Linear acc.", "MLP acc.", "Delay (ms)", "Within 50 ms", "Within 100 ms", "Payload false positives", "Blocked: MLP acc.", "Blocked delay (ms)"],
                 rows, caption=caption or f"Cue audit on episodes of the {LABELS.get('pd_residual_acmed_s02_lowrate') if controller == 'residual' else 'PD'}: probes from candidate input windows (built as a policy holds them) to the contact state, fitted on the even cases of the training-split surface families and of payload-carrying free motion, tested on the odd ones. Balanced accuracies of a linear and a 64-unit MLP probe; median detection delay after an onset and the fraction of onsets detected within 50 and 100 ms (MLP); false-positive rate on payload-only free motion (MLP); the same for the blocked state. Source: {path}.",
                 numeric_cols=tuple(range(1, 10)))


def sensing_table(path: Path | list[Path], names: dict[str, str] | None = None, caption: str | None = None) -> str:
    paths = [p for p in (path if isinstance(path, list) else [path]) if p.exists()]
    if not paths:
        return ""
    d = {}
    for p in paths:
        d.update(json.loads(p.read_text()))
    path = ", ".join(str(p) for p in paths)
    rows = []
    for name, r in d.items():
        rows.append([(names or {}).get(name, name), f"{r['probe_balanced_accuracy']:.2f}", f"{1e3 * r['detection_delay_median_s']:.0f}", f"{r.get('within_50ms', float('nan')):.2f}", f"{r.get('within_100ms', float('nan')):.2f}",
                     f"{r.get('blocked_probe_balanced_accuracy', float('nan')):.2f}", f"{1e3 * r.get('blocked_detection_delay_median_s', float('nan')):.0f}",
                     f"{r['kp_scale_in_contact_mean']:.2f} / {r['kp_scale_free_mean']:.2f}" if "kp_scale_in_contact_mean" in r else "-",
                     f"{r['residual_normal_force_in_contact_N']:+.1f} / {r['residual_normal_force_free_N']:+.1f}" if "residual_normal_force_in_contact_N" in r else "-"])
    return table(["Controller", "Contact: linear acc.", "Delay (ms)", "Within 50 ms", "Within 100 ms", "Blocked: acc.", "Blocked delay (ms)", "Stiffness multiplier, contact / free", "Residual normal force, contact / free (N)"],
                 rows, caption=caption or f"Contact sensing of the learned controllers: a class-balanced linear probe from the trunk features to the contact state and to the blocked state on 8 validation cases per surface family (fitted on even cases, tested on odd ones): balanced accuracy, median detection delay after an onset, fraction of onsets detected within 50 and 100 ms; and what the output does in contact (mean stiffness multiplier; mean residual torque mapped to a tool force along the surface normal, positive away from the surface). Source: {path}.", numeric_cols=(1, 2, 3, 4, 5, 6))


def training_curves(runs: list[str], x_key: str, x_label: str, name: str, caption: str, fig_id: str) -> str:
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    seen = set()
    for run in runs:
        rd = Path(run)
        if not (rd / "log.jsonl").exists():
            continue
        recs = [json.loads(l) for l in open(rd / "log.jsonl")]
        vals = [r for r in recs if r.get("val")]
        key = json.loads((rd / "meta.json").read_text()).get("method_key", rd.name)
        lab = label(key) if key not in seen else None; seen.add(key)
        col = color(key)
        x = [r[x_key] + 1 for r in vals]
        axes[0].plot(x, [r["val"]["mean"] for r in vals], marker="o", ms=2.5, lw=0.8, color=col, label=lab)
        axes[1].plot(x, [r["val"]["per_regime"]["blocked"] for r in vals], marker="o", ms=2.5, lw=0.8, color=col)
        axes[2].plot(x, [r["val"]["excess_force_cap"] for r in vals], marker="o", ms=2.5, lw=0.8, color=col)
        axes[3].plot(x, [np.mean([r["val"]["per_regime"][k] for k in ("hold", "free", "policy", "step")]) for r in vals], marker="o", ms=2.5, lw=0.8, color=col)
    ref = Path("runs/d6/val_pd_shared.json")
    if ref.exists():
        pd = json.loads(ref.read_text())
        axes[0].axhline(pd["mean"], color="#8c8c8c", lw=0.8, ls=(0, (4, 2)), label="PD, shared gains, same cases")
        axes[1].axhline(pd["per_regime"]["blocked"], color="#8c8c8c", lw=0.8, ls=(0, (4, 2)))
        axes[2].axhline(pd["excess_force_cap"], color="#8c8c8c", lw=0.8, ls=(0, (4, 2)))
        axes[3].axhline(np.mean([pd["per_regime"][k] for k in ("hold", "free", "policy", "step")]), color="#8c8c8c", lw=0.8, ls=(0, (4, 2)))
    for ax, yl in zip(axes, ("validation composite (cap), all families", "blocked composite (cap)", "excess force above 15 N (N)", "free-space composite (mean of 4 families)")):
        ax.set_ylabel(yl); ax.set_xlabel(x_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight"); plt.close(fig)
    return figure(f"figures/{name}.svg", caption, fig_id, wide=True)
