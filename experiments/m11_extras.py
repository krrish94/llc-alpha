"""Milestone 11 report extras (ADR 0011): the interface ablation across profiles, the torque-tracking family, the
period and delay sweep, the effective stiffness, the pick family with a grasp, the learned-interface runs, the error
floor and the probing study. Every function reads the metrics files under runs/m11 and returns HTML or writes a figure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from report_api import figure, table  # noqa: E402

from experiments.d7_extras import LABELS, label, load, mean_metric  # noqa: E402
from experiments.d_report import GROUP_LABELS, REGIME_SHORT, add_best_classical, pooled  # noqa: E402

RUNS = Path("runs/m11")
REPORT = Path("research/reports/19-m11-interfaces")
FIG = REPORT / "figures"
PROFILE_LABELS = {"p0": "P0 torque, 1 kHz", "p1": "P1 MIT mode, 250 Hz", "p2": "P2 goal position, RAM gains, 100 Hz", "p3": "P3 goal position, fixed gains, 50 Hz"}
LABELS.update({"torque_program": "Torque program on the feedforward PD", "oracle_pd": "Oracle around the PD (privileged ceiling)", "best_classical": "Best classical controller of the family",
               "pd_shared": "PD tuned on the tracking composite", "pd_shared_cap": "PD tuned on the cap composite", "pd_feedforward": "PD (tracking-tuned) + inverse-dynamics feedforward",
               "pd_shared_cap_feedforward": "PD (cap-tuned) + inverse-dynamics feedforward", "distill_v3_r2_two_teachers": "Two-teacher student (Milestone 10, round 2)"})
FAMILIES = ("hold", "free", "policy", "step", "contact2", "blocked", "task", "push", "pick")


def load_profiles(names=("p0", "p1", "p2", "p3")) -> dict[str, dict]:
    """{profile: {group: pooled results with best_classical}} from runs/m11/metrics_<profile>.json."""
    out = {}
    for p in names:
        path = RUNS / f"metrics_{p}.json"
        if not path.exists():
            continue
        data = json.load(open(path))
        groups = {}
        for g, res in data["groups"].items():
            groups[g], _ = pooled(res, data["learned"])
            add_best_classical(groups[g])
        out[p] = groups
    return out


def profile_table(profiles: dict, method: str = "best_classical", groups=("in_distribution", "tasks", "surface_500"), caption: str | None = None) -> str:
    """Rows: group and family; columns: the profiles; cells: the method's cap composite with its ratio to P0 (hypothesis P1)."""
    names = [p for p in ("p0", "p1", "p2", "p3") if p in profiles]
    rows = []
    for g in groups:
        if not all(g in profiles[p] for p in names):
            continue
        regimes = [r for r in FAMILIES if any(m["regime"] == r for m in profiles[names[0]][g].get(method, []))]
        for r in regimes:
            row = [GROUP_LABELS.get(g, g), REGIME_SHORT.get(r, r)]
            base = mean_metric(profiles["p0"][g][method], r, "composite_cap") if "p0" in profiles and method in profiles["p0"][g] else float("nan")
            for p in names:
                if method not in profiles[p][g]:
                    row.append("—"); continue
                v = mean_metric(profiles[p][g][method], r, "composite_cap")
                src = ""
                if method == "best_classical":
                    src = next(m["best_classical_source"] for m in profiles[p][g][method] if m["regime"] == r)
                    src = {"pd_shared": "PD", "pd_shared_cap": "PD cap", "pd_feedforward": "PD+ff", "pd_shared_cap_feedforward": "PD cap+ff"}.get(src, src)
                row.append(f"{v:.2f}" + (f" ({v / base:.2f}×)" if p != "p0" and np.isfinite(base) and base > 0 else "") + (f" [{src}]" if src else ""))
            rows.append(row)
    return table(["Group", "Family"] + [PROFILE_LABELS[p] for p in names], rows, caption=caption or f"{label(method)}: cap composite per family through each interface profile with the model-based interface (mean over 16 cases; in brackets the ratio to the torque interface; the best classical controller of each family in square brackets).", css_class="wide")


def torque_table(profiles: dict, caption: str | None = None) -> str:
    """The torque-tracking family per profile and program kind: realised-torque error, delay, bandwidth, saturation and torque rate."""
    names = [p for p in ("p0", "p1", "p2", "p3") if p in profiles and "torque_tracking" in profiles[p]]
    kinds = ("hold", "step", "chirp", "torque_chirp")
    rows = []
    for p in names:
        res = profiles[p]["torque_tracking"]["torque_program"]
        for kind in kinds:
            sub = [m for i, m in enumerate(res) if ("hold", "step", "chirp", "torque_chirp")[i % 4] == kind]
            if not sub:
                continue
            f = lambda k: float(np.nanmean([m[k] for m in sub]))
            rows.append([p.upper(), {"hold": "hold", "step": "steps", "chirp": "chirp", "torque_chirp": "pure-torque chirp"}[kind],
                         f"{100 * f('tau_track_rmse'):.2f}", f"{1e3 * f('tau_track_delay'):.0f}" if np.isfinite(f("tau_track_delay")) else "—", f"{f('tau_track_bandwidth'):.0f}" if np.isfinite(f("tau_track_bandwidth")) else "—",
                         f"{100 * f('saturation'):.1f}", f"{f('torque_rate_rms'):.0f}", f"{f('composite_cap'):.2f}"])
    return table(["Profile", "Program", "Torque error (% of stall)", "Delay (ms)", "Bandwidth (Hz)", "Saturation (%)", "Torque rate (Nm/s)", "Composite"], rows,
                 caption=caption or "Torque-tracking family (protocol v4): the realised joint torque against the upper command evaluated at the true state at every physics step (RMS as a fraction of the stall torque), the feedforward PD plus a torque program through the model-based interface of each profile (P0 torque at 1 kHz; P1 MIT mode at 250 Hz; P2 goal position with RAM gains at 100 Hz; P3 goal position with fixed gains at 50 Hz); four cases per program kind: the PD frame alone (hold), steps of 10 to 30% of the stall torque on one joint, a chirp from 0.5 to 125 Hz at 10%, and the chirp as a pure torque command at 2%. The delay is the cross-correlation lag of the programmed joint; the bandwidth the chirp frequency at which the realised amplitude falls below 0.7 of the command.", numeric_cols=(2, 3, 4, 5, 6, 7))


def stiffness_table(profiles: dict, methods=("pd_feedforward", "oracle_pd"), caption: str | None = None) -> str:
    """Blocked family: the contact stiffness realised through each profile (the realised fraction of the commanded impedance)."""
    names = [p for p in ("p0", "p1", "p2", "p3") if p in profiles]
    rows = []
    for k in methods:
        base = mean_metric(profiles["p0"]["in_distribution"][k], "blocked", "contact_impedance") if "p0" in profiles else float("nan")
        row = [label(k)]
        for p in names:
            v = mean_metric(profiles[p]["in_distribution"][k], "blocked", "contact_impedance")
            exc = mean_metric(profiles[p]["in_distribution"][k], "blocked", "excess_force_cap")
            row.append(f"{v:.0f} ({v / base:.2f}×), {exc:.1f} N above the cap")
        rows.append(row)
    return table(["Controller"] + [PROFILE_LABELS[p] for p in names], rows, caption=caption or "Blocked family: contact stiffness (N/m, mean normal force per metre of actual tool error while the reference is inside the surface) and the excess force above the 15 N cap, per profile; the ratio to the torque interface is the realised fraction of the commanded impedance.", css_class="wide")


def sweep_figure(fig_id: str = "fig-sweep") -> str:
    """Cap composites of the feedforward PD and the torque error of the program over the host period (2 ms delay) and over the bus delay (10 ms period), P2-like interface."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plotstyle import apply_style, design

    apply_style()
    per = {}; dly = {}
    for path in RUNS.glob("sweep_period_*.json"):
        per[float(path.stem.split("_")[-1])] = json.load(open(path))
    for path in RUNS.glob("sweep_delay_*.json"):
        dly[float(path.stem.split("_")[-1])] = json.load(open(path))
    base = RUNS / "metrics_p2.json"
    if base.exists():  # the profile's own point (10 ms, 2 ms)
        d = json.load(open(base)); per.setdefault(0.01, d); dly.setdefault(0.002, d)
    if not per or not dly:
        return ""
    fams = ("hold", "free", "policy", "step")
    colors = design()["methods"]
    palette = [colors[k] for k in ("pd_shared", "pd_residual", "learned_gains", "neural_torque")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
    for ax, data, key, xlabel in ((axes[0], per, "period", "Host period (ms)"), (axes[1], dly, "delay", "Bus delay (ms)")):
        xs = sorted(data)
        for fam, c in zip(fams, palette):
            ys = [np.mean([m["composite_cap"] for m in data[x]["groups"]["in_distribution"]["pd_feedforward"] if m["regime"] == fam]) for x in xs]
            ax.plot([1e3 * x for x in xs], ys, "o-", color=c, label=REGIME_SHORT[fam])
        ax.set_xlabel(xlabel); ax.set_ylabel("Cap composite, feedforward PD")
        if key == "period":
            from matplotlib.ticker import NullFormatter

            ax.set_xscale("log"); ax.set_xticks([2, 4, 10, 20, 40]); ax.set_xticklabels(["2", "4", "10", "20", "40"]); ax.xaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.subplots_adjust(top=0.86, wspace=0.35)
    ax = axes[2]
    for data, key, c, lab in ((per, "period", colors["pd_shared"], "over the period (2 ms delay)"), (dly, "delay", colors["pd_residual"], "over the delay (10 ms period)")):
        xs = sorted(data)
        ys = [100 * np.mean([m["tau_track_rmse"] for m in data[x]["groups"]["torque_tracking"]["torque_program"]]) for x in xs]
        ax.plot([1e3 * x for x in xs], ys, "o-", color=c, label=lab)
    ax.set_xlabel("Period or delay (ms)"); ax.set_ylabel("Torque error (% of stall)"); ax.legend(loc="lower right")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "sweep.svg"); plt.close(fig)
    return figure("figures/sweep.svg", "Hypothesis P1, the descriptive curve: the feedforward PD through a goal-position interface with RAM gains (the P2 structure) over the host period at a 2 ms bus delay (left) and over the bus delay at a 10 ms period (middle), cap composite per free-space family (16 cases each); right, the torque-tracking family's realised-torque error over the same sweeps.", fig_id, wide=True)


def pick_table(path: Path = RUNS / "metrics_tasks_p0.json", caption: str | None = None) -> str:
    """The pick family with a grasp (protocol v4) on the torque interface: grasp success, pre-grasp disturbance, grasp force, place error, task success."""
    if not path.exists():
        return ""
    data = json.load(open(path))
    res, _ = pooled(data["groups"]["tasks"], data["learned"])
    rows = []
    short = {"pd_shared": "PD, tracking-tuned", "pd_shared_cap": "PD, cap-tuned", "pd_feedforward": "PD, tracking-tuned, + feedforward", "pd_shared_cap_feedforward": "PD, cap-tuned, + feedforward", "oracle_pd": "Oracle (ceiling)", "distill_v3_r2_two_teachers": "Two-teacher student (3 seeds)"}
    for k in ("pd_shared", "pd_shared_cap", "pd_feedforward", "pd_shared_cap_feedforward", "oracle_pd", "distill_v3_r2_two_teachers"):
        if k not in res:
            continue
        pick = [m for m in res[k] if m["regime"] == "pick"]; push = [m for m in res[k] if m["regime"] == "push"]
        f = lambda ms, key: float(np.nanmean([m[key] for m in ms]))
        rows.append([short.get(k, label(k)), f"{100 * f(pick, 'grasp_success'):.0f}", f"{1e3 * f(pick, 'pregrasp_disturbance'):.1f}", f"{f(pick, 'grasp_force_peak_20ms'):.1f}", f"{1e3 * f(pick, 'place_error'):.0f}", f"{f(pick, 'place_force_peak_20ms'):.1f}",
                     f"{100 * f(pick, 'task_success'):.0f}", f"{100 * f(push, 'task_success'):.0f}"])
    return table(["Controller", "Grasp (%)", "Pre-grasp disturbance (mm)", "Grasp force (N)", "Place error (mm)", "Place force (N)", "Pick success (%)", "Push success (%)"], rows, css_class="wide",
                 caption=caption or "Pick-and-place with a grasp (protocol v4, torque interface, 16 cases with payloads of 0, 1 and 2 kg at the tool): the weld activates only if the block's centre is within 6 mm laterally and 9 mm along the tool axis at the grasp instant. The students are pooled over three seeds.", numeric_cols=(1, 2, 3, 4, 5, 6, 7))


def interface_table(caption: str | None = None) -> str:
    """Learned interface per profile (Phase 2a): the model-based warm start and the best checkpoint on the validation cases."""
    rows = []
    for p in ("p1", "p2", "p3"):
        for suffix, lab in (("", "rate weight 3e-4"), ("_ratew", "rate weight 3e-3")):
            path = RUNS / f"iface_{p}{suffix}-s0" / "log.jsonl"
            if not path.exists():
                continue
            recs = [json.loads(l) for l in open(path)]
            vals = [r for r in recs if "val" in r]
            if not vals:
                continue
            warm = vals[0]["val"]; best = min(vals, key=lambda r: r["val"]["mean"])["val"]
            f = lambda v: f"{v['mean']:.2f}; {100 * v['tau_track_rmse']:.2f}% / {1e3 * v['tau_track_delay']:.0f} ms / {v['torque_rate']:.0f} Nm/s"
            rows.append([PROFILE_LABELS[p], lab, f(warm), f(best), f"{best['tau_track_rmse'] / warm['tau_track_rmse']:.2f}", f"{best['torque_rate'] / warm['torque_rate']:.2f}"])
    if not rows:
        return ""
    return table(["Profile", "Training", "Model-based interface: validation mean; torque error / delay / rate", "Best learned checkpoint", "Error ratio", "Rate ratio"], rows,
                 caption=caption or "Learned interface (Phase 2a) on the validation cases (8 torque-tracking, hold, free and blocked under the feedforward PD): the mean cap composite and the torque-tracking metrics of the model-based warm start and of the best checkpoint by validation; the pre-registered criterion asks for an error ratio of 0.5 at a rate ratio below 1.5.", css_class="wide")


def floor_table(path: Path = RUNS / "floor_hold.json", caption: str | None = None) -> str:
    if not path.exists():
        return ""
    d = json.load(open(path))
    variants = {"as_is": "as is", "exact_velocity": "exact velocity feedback", "exact_feedback": "exact position and velocity feedback", "no_delay": "no bus delay", "period_1ms": "1 ms host period", "period_1ms_no_delay": "1 ms period, no delay", "all_removed": "all removed"}
    rows = []
    for v, lab in variants.items():
        rows.append([lab] + [f"{100 * d['profiles'][p][v]['tau_track_rmse']:.2f}" if v in d["profiles"].get(p, {}) else "—" for p in ("p1", "p2", "p3")])
    return table(["Model-based interface with"] + [PROFILE_LABELS[p] for p in ("p1", "p2", "p3")], rows,
                 caption=caption or f"Error floor of the model-based interface on {d['batch']} fixed hold windows with torque programs (realised-torque error, % of stall, RMS): the profile as is, and with one of its limits removed at a time. What remains with exact feedback, a 1 ms period and no delay is the part a correction on the profile's channels could remove; the rest is the servo acting on its own quantised measurement between host frames.", numeric_cols=(1, 2, 3))


PROBE_COLS_EXT = [("contact_mlp_balanced_accuracy", "contact, MLP probe (bal. acc.)"), ("normal_mlp_r2", "normal, MLP probe (R²)"), ("depth_r2", "reference depth in the surface (R²)"), ("depth_mlp_r2", "depth, MLP probe (R²)"),
                  ("sliding_balanced_accuracy", "sliding vs pressing (bal. acc.)"), ("blocked_phase_balanced_accuracy", "blocked phase (bal. acc.)"), ("release_phase_balanced_accuracy", "release phase (bal. acc.)"),
                  ("in_gap_balanced_accuracy", "block in the gap before the grasp (bal. acc.)")]


def probe_table(path: Path = RUNS / "probe_panda.json", caption: str | None = None, extended: bool = False) -> str:
    if not path.exists():
        return ""
    d = json.load(open(path))
    cols = [("contact_balanced_accuracy", "contact (bal. acc.)"), ("contact_within_50ms", "detected within 50 ms"), ("wall_balanced_accuracy", "table vs wall (bal. acc.)"), ("normal_r2", "normal in the tool frame (R²)"),
            ("stiff_2000_balanced_accuracy", "2 kN/m surface (bal. acc.)"), ("stiff_500_balanced_accuracy", "500 N/m surface (bal. acc.)"), ("payload_r2", "payload mass (R²)"), ("f_ext_r2", "external force (R²)"), ("f_normal_r2", "contact force (R²)")]
    if extended:
        cols = PROBE_COLS_EXT
    rows = []
    present = [(k, c) for k, c in cols if any(k in rec[src] and np.isfinite(rec[src][k]) for rec in d.values() for src in rec if isinstance(rec[src], dict))]
    cols = present
    for name, rec in d.items():
        for src in ("input", "layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8"):
            if src not in rec:
                continue
            rows.append([name, {"input": "observation"}.get(src, src)] + [f"{rec[src][k]:.2f}" if k in rec[src] and np.isfinite(rec[src][k]) else "—" for k, _ in cols])
    if not cols:
        return ""
    return table(["Controller", "Probe from"] + [c for _, c in cols], rows, caption=caption or ("Extended probing targets (ADR 0011 amendment 3): a nonlinear probe (one hidden layer of 64) beside the linear one for contact and the normal; the reference's depth inside the surface (the quantity the impedance law needs, among contact decisions); sliding against pressing (tool tangential speed above 2 cm/s, among contact decisions); the blocked and release phases of a blocked episode (among its decisions); and whether the block's centre is inside the grasp tolerance before the grasp instant (pick episodes). Balanced accuracy for classes (chance 0.5), R² for regressions." if extended else "Probing study: linear probes fitted on even validation episodes and tested on odd ones (surface, free-space, compliant-surface and payload cases), from the observation and from every trunk layer of the two-teacher student (seeds 0 and 1). Balanced accuracy for classes (chance 0.5), R² for regressions (0 is the mean predictor; negative is worse)."), css_class="wide")


def nad_table(path: Path = RUNS / "nad_fit.json", caption: str | None = None) -> str:
    """Phase 0: what the Neural Actuation Dataset identifies of the two servo classes (per joint: the command register's
    gain on the position error, the DC-motor law's resistance and back-EMF constant with the datasheet cross-checks)."""
    if not path.exists():
        return ""
    d = json.load(open(path))
    rows = []
    for j in d["xm430"]["joints"]:
        rows.append(["XM430-W350 (OpenManipulator-X)", str(j["joint"]), f"{j['pwm_per_rad']:.0f} PWM/rad (R² {j['pwm_fit_r2']:.2f})", f"{j['R_ohm']:.2f}", f"{j['KeN_Vs_per_rad']:.2f}", f"{j['current_fit_r2']:.2f}",
                     f"{j['stall_current_A_at_12V']:.2f} A / {j['no_load_rpm_at_12V']:.0f} rpm", f"{1e3 * j['velocity_step_rad_s']:.0f} mrad/s"])
    for j in d["sts3215"]["joints"]:
        rows.append(["STS3215 (SO-101)", str(j["joint"]), f"{-j['load_per_rad']:.0f} ‰ duty/rad (R² {j['load_fit_r2']:.3f})", "—", "—", "—", "—", f"{j['velocity_step_steps_s']:.0f} steps/s ({1e3 * j['velocity_step_steps_s'] * 2 * np.pi / 4096:.0f} mrad/s)"])
    return table(["Servo", "Joint", "Command register on the position error", "R (Ω)", "K_e N (V s/rad)", "Current fit R²", "Implied stall current / no-load speed at 12 V (datasheet 2.3 A / 46 rpm)", "Velocity feedback step"], rows,
                 caption=caption or "Phase 0, identification on the Neural Actuation Dataset (training split; 280 OpenManipulator-X and 80 SO-101 trajectories at 59 and 62 Hz). The XM430's current follows the DC-motor law i = (d V - K_e N ω) / R with R and K_e N consistent with the datasheet's stall current and no-load speed on joints 2 to 5 (joint 1, the base yaw, moves little and its fit is poor); its PWM register is not the firmware's P term alone (the integral and the derivative on an internal velocity estimate enter), so the position loop's gains are taken from the e-manual's conversion and randomised in simulation. The STS3215's load register is the P term on the position error with no visible velocity term. Both report velocity in coarse steps, the resolution the servo profiles P2 and P3 carry.", css_class="wide")


def retune_table(caption: str | None = None) -> str:
    """The PD re-tuned per profile (phase1_tune.sh, CMA-ES through the profile's servo loop on the cap composite) against the
    torque-interface gains through the same profile (ADR 0011 amendment 3, item 1)."""
    fams = ("hold", "free", "policy", "step", "contact2", "blocked")
    rows = []
    for p in ("p1", "p2", "p3"):
        base_path, tuned_path = RUNS / f"metrics_{p}.json", RUNS / f"metrics_tune_{p}.json"
        if not (base_path.exists() and tuned_path.exists()):
            continue
        kp0 = None
        for name, path, key in (("torque-interface gains (PD+ff)", base_path, "pd_feedforward"), ("re-tuned through the profile (PD+ff)", tuned_path, "pd_shared")):
            data, groups = load(path)
            res = groups["in_distribution"][key]
            kp = np.asarray(data["controllers"][key]["inner"]["kp"], float)
            kp0 = kp if kp0 is None else kp0
            task = groups.get("tasks", {}).get(key, [])
            rows.append([PROFILE_LABELS[p], name, f"{np.exp(np.mean(np.log(kp / kp0))):.2f}×"] + [f"{mean_metric(res, r, 'composite_cap'):.2f}" for r in fams]
                        + [f"{mean_metric(task, 'push', 'composite_cap'):.2f} / {mean_metric(task, 'pick', 'composite_cap'):.2f}" if task else "—", f"{1e3 * mean_metric(res, 'free', 'pos_rmse'):.1f}", f"{mean_metric(res, 'free', 'torque_rate_rms'):.0f}",
                           f"{mean_metric(res, 'blocked', 'excess_force_cap'):.1f}", f"{mean_metric(res, 'blocked', 'contact_impedance'):.0f}"])
    if not rows:
        return ""
    return table(["Profile", "Gains", "Stiffness (geometric mean ratio)"] + [REGIME_SHORT.get(r, r) for r in fams] + ["Push / pick", "Free-motion RMSE (mrad)", "Free-motion torque rate (Nm/s)", "Blocked excess above the cap (N)", "Blocked contact stiffness (N/m)"], rows,
                 caption=caption or "The feedforward PD with the torque-interface gains and with the gains re-tuned through each profile (CMA-ES on the cap composite, 40 iterations of 12, the Milestone 9 budget), both through the profile's servo loop with the model-based interface and the servo's nominal spring set to the same gains; cap composites per family (in distribution, 16 cases each) and the free-motion and blocked columns the criterion turns on.", css_class="wide")


def velfilter_table(caption: str | None = None) -> str:
    """The velocity-feedback filter sweep through P1 (ADR 0011 amendment 3, item 3): the classical feedforward PD, the oracle and
    the host-rate student with the servo damping on the raw quantised velocity and on a filtered estimate."""
    files = [("raw", RUNS / "metrics_students_p1.json")] + [(f"{1e3 * v:.0f} ms", RUNS / f"velfilter_p1_{v}.json") for v in (0.002, 0.005, 0.01)]
    keys = (("pd_feedforward", "PD + feedforward"), ("oracle_pd", "Oracle around the PD"), ("distill_host_p1", "Host-rate student"))
    rows = []
    for lab, path in files:
        if not path.exists():
            continue
        data, groups = load(path)
        res = groups["in_distribution"]
        tt = groups.get("torque_tracking", {}).get("torque_program", [])
        for key, name in keys:
            if key not in res:
                continue
            r = res[key]
            rows.append([lab, name, f"{mean_metric(r, 'free', 'torque_rate_rms'):.0f}", f"{mean_metric(r, 'policy', 'torque_rate_rms'):.0f}", f"{1e3 * mean_metric(r, 'free', 'pos_rmse'):.1f}",
                         f"{mean_metric(r, 'hold', 'composite_cap'):.2f}", f"{mean_metric(r, 'free', 'composite_cap'):.2f}", f"{mean_metric(r, 'policy', 'composite_cap'):.2f}", f"{mean_metric(r, 'step', 'composite_cap'):.2f}",
                         f"{mean_metric(r, 'blocked', 'excess_force_cap'):.1f}", f"{mean_metric(r, 'blocked', 'composite_cap'):.2f}", f"{mean_metric(r, 'contact2', 'composite_cap'):.2f}",
                         (f"{100 * float(np.nanmean([m['tau_track_rmse'] for m in tt])):.2f}" if (tt and key == "pd_feedforward") else "—")])
    if not rows:
        return ""
    return table(["Velocity filter", "Controller", "Torque rate, free motion (Nm/s)", "Torque rate, policy-like (Nm/s)", "Free-motion RMSE (mrad)", "Hold", "Free", "Policy", "Step", "Blocked excess (N)", "Blocked", "Press-slide", "Torque error (% of stall)"], rows,
                 caption=caption or "Through P1 with the servo damping on the raw quantised velocity feedback (raw) and on a first-order filtered estimate (time constant 2, 5 and 10 ms; the host sees the same estimate): torque rate and tracking on the free-space families, the blocked family's excess force above the 15 N cap, and the cap composites (in distribution, 16 cases per family); the torque-tracking error of the feedforward PD with the torque program (mean over the four program kinds).", css_class="wide")


def motor_table(without: Path, with_motor: Path, keys, caption: str | None = None, groups=("in_distribution", "surface_500", "tasks", "payload_1.5kg")) -> str:
    """The same controllers without and with the DC-motor layer (ADR 0011 amendment 3, item 5): cap composites per family and
    group, and the student's criterion count under the layer."""
    if not (without.exists() and with_motor.exists()):
        return ""
    _, a = load(without); _, b = load(with_motor)
    for g in a.values():
        add_best_classical(g)
    for g in b.values():
        add_best_classical(g)
    rows = []
    for g in groups:
        if g not in b:
            continue
        regimes = [r for r in FAMILIES if any(m["regime"] == r for m in next(iter(b[g].values())))]
        for r in regimes:
            row = [GROUP_LABELS.get(g, g), REGIME_SHORT.get(r, r)]
            for k in keys:
                if k not in b[g]:
                    row.append("—"); continue
                va, vb = mean_metric(a[g][k], r, "composite_cap") if k in a.get(g, {}) else float("nan"), mean_metric(b[g][k], r, "composite_cap")
                row.append(f"{va:.2f} → {vb:.2f}")
            rows.append(row)
    return table(["Group", "Family"] + [label(k) for k in keys], rows, caption=caption or "Cap composites without → with the DC-motor layer under the servo law (the layer's unknowns drawn per case: torque-constant error ±15%, voltage envelope, viscous and Coulomb friction, current-loop lag up to 2 ms, torque-feedback resolution); the same cases, the same controllers through P1.", css_class="wide")


def variants_table(files, caption: str | None = None) -> str:
    """Recipe variants of the P1 host-rate student (Phase 2c and 2d, ADR 0011 amendment 3): one row per learned run found in the
    files, its recipe from the run's meta, the selected iteration, the in-distribution composites of the families the criterion
    turns on, the blocked excess force, the press force on the sliding family, and the criterion count over every group."""
    rows = []
    seen = set()
    for path in files:
        path = Path(path)
        if not path.exists():
            continue
        data, groups = load(path)
        for g in groups.values():
            add_best_classical(g)
        res = groups["in_distribution"]
        by_key: dict[str, list] = {}
        for name, info in data["learned"].items():
            by_key.setdefault(info.get("method_key") or name, []).append(info)
        for mk, infos in by_key.items():
            if mk not in res or mk in seen:
                continue
            seen.add(mk)
            runs = [Path(info["path"]).parent for info in infos]
            a = json.load(open(runs[0] / "meta.json"))["args"]
            bests = [json.load(open(r / "best.json")) if (r / "best.json").exists() else {} for r in runs]
            seeds = sorted(json.load(open(r / "meta.json"))["args"].get("seed", 0) for r in runs)
            best = {"iter": ", ".join(str(b.get("iter", "—")) for b in bests), "val": float(np.mean([b.get("val", float("nan")) for b in bests]))}
            recipe = f"free {a.get('free_weight', 1):g}, contact {a.get('contact_weight', 1):g}, teacher share {a.get('betas', '')}, force term {a.get('force_weight', 0):g}, {a.get('layers', 2)} layers, " + (f"seeds {', '.join(map(str, seeds))} (pooled)" if len(seeds) > 1 else f"seed {seeds[0]}")
            n_ok = n_tot = 0
            for gg in groups.values():
                if mk not in gg or "best_classical" not in gg:
                    continue
                for reg in sorted({m["regime"] for m in gg[mk]}):
                    v, b = mean_metric(gg[mk], reg, "composite_cap"), mean_metric(gg["best_classical"], reg, "composite_cap")
                    if np.isfinite(v) and np.isfinite(b):
                        n_tot += 1; n_ok += int(v <= 1.05 * b)
            r = res[mk]
            rows.append([recipe, f"{best['iter']}", f"{best['val']:.2f}"] + [f"{mean_metric(r, reg, 'composite_cap'):.2f}" for reg in ("hold", "free", "policy", "step", "contact2", "blocked")]
                        + [f"{mean_metric(r, 'blocked', 'excess_force_cap'):.1f}", f"{mean_metric(r, 'contact2', 'f_mean'):.1f}", f"{mean_metric(r, 'free', 'torque_rate_rms'):.0f}", f"{n_ok} / {n_tot}"])
        if "oracle_pd" in res and "oracle" not in seen:
            seen.add("oracle"); r = res["oracle_pd"]
            rows.append(["Oracle around the PD through P1 (the teacher in contact)", "—", "—"] + [f"{mean_metric(r, reg, 'composite_cap'):.2f}" for reg in ("hold", "free", "policy", "step", "contact2", "blocked")]
                        + [f"{mean_metric(r, 'blocked', 'excess_force_cap'):.1f}", f"{mean_metric(r, 'contact2', 'f_mean'):.1f}", f"{mean_metric(r, 'free', 'torque_rate_rms'):.0f}", "—"])
    if not rows:
        return ""
    return table(["Recipe", "Selected iteration", "Validation", "Hold", "Free", "Policy", "Step", "Press-slide", "Blocked", "Blocked excess above the cap (N)", "Press force, sliding family (N)", "Free-motion torque rate (Nm/s)", "Criterion rows within 5%"], rows,
                 caption=caption or "The P1 host-rate student's recipe varied (ADR 0011 amendment 3, item 2; one seed each): the label weights of the free-space and contact rows, the teacher's share of the collection episodes per iteration (the last value repeats), and the force-space label term (the force along the surface normal the torque error implies, per the 15 N cap). Checkpoints selected on the 32 validation cases; in-distribution cap composites (16 cases per family); the criterion count over every group and family against the best classical controller through P1, the re-tuned PD included.", css_class="wide")


def diag_table(path: Path = RUNS / "blocked_diag_p1.json", caption: str | None = None) -> str:
    """The label-fit diagnosis of a student along its own trajectories (experiments/m11_blocked_diag.py)."""
    if not path.exists():
        return ""
    d = json.load(open(path))
    rows = []
    keys = (("blocked/approach", "blocked, approach"), ("blocked/blocked", "blocked, reference inside"), ("blocked/release", "blocked, release"), ("contact2/press", "press-slide, press"), ("contact2/slide", "press-slide, slide"), ("hold", "hold"), ("free", "free motion"))
    for ckpt, rec in d["checkpoints"].items():
        for key, lab in keys:
            s = rec["per_phase"].get(key) or rec["per_family"].get(key)
            if not s or "fit_rms_nm" not in s:
                continue
            rows.append([f"iteration {rec['iter']}" if rec["iter"] is not None and rec["iter"] >= 0 else ckpt, lab, f"{s['n']}", f"{s['fit_rms_nm']:.2f}", f"{s['yield_teacher_rms_nm']:.1f} / {s['yield_student_rms_nm']:.1f}", f"{s['yield_corr']:.2f}",
                         f"{s['kp_scale_mean']:.2f} / {s['kd_scale_mean']:.2f}" if "kp_scale_mean" in s else "—", f"{s['residual_rms_nm']:.2f}" if "residual_rms_nm" in s else "—",
                         f"{s['push_error_mean_n']:+.1f} ({s['push_error_rms_n']:.1f})" if "push_error_mean_n" in s else "—"])
    return table(["Checkpoint", "Rows", "Decisions", "Label fit, RMS (Nm)", "Yield from the PD base, teacher / student, RMS (Nm)", "Yield correlation", "Gain multipliers kp / kd", "Residual, RMS (Nm)", "Push error along the normal, mean (RMS) (N)"], rows,
                 caption=caption or "Teacher labels collected along the student's own episodes through P1 (eight train cases per family): the RMS error between the student's fast-loop torque and the teacher's, the yield of each from the feedforward PD base, the student's gain multipliers and residual, and the force along the surface normal that the torque error implies (positive: the student pushes harder than the teacher), for the checkpoint after the teacher-driven iteration and the selected one.", css_class="wide")


def transient_table(path: Path = RUNS / "blocked_transient_p1.json", cap: float = 15.0, caption: str | None = None) -> str:
    """Where the blocked excess force falls in time (experiments/m11_blocked_transient.py)."""
    if not path.exists():
        return ""
    d = json.load(open(path))
    rows = []
    for name, rec in d.items():
        a = rec["aggregate"]
        pls = sorted({p["payload"] for p in rec["per_case"]})
        rows.append([label(name) if name in LABELS else name, f"{a['mean']:.1f}", f"{a['peak']:.0f}", f"{a['first_0.5s_mean']:.1f} / {a['middle_mean']:.1f} / {a['last_0.5s_mean']:.1f}", f"{a['excess_mean']:.1f}",
                     f"{100 * a['first_0.5s_excess_share']:.0f} / {100 * a['middle_excess_share']:.0f} / {100 * a['last_0.5s_excess_share']:.0f}", f"{a['settle_under_cap_s']:.2f}",
                     " / ".join(f"{a[f'excess_mean_payload_{pl:g}']:.1f}" for pl in pls)])
    return table(["Controller", "Mean force in the blocked phase (N)", "Peak (N)", "Mean force, first 0.5 s / middle / last 0.5 s (N)", f"Excess above {cap:g} N (N)", "Share of the excess, first 0.5 s / middle / last (%)", "Time to settle under the cap (s)", "Excess by payload, 0 / 1 / 2 kg (N)"], rows,
                 caption=caption or "The sixteen in-distribution blocked test cases through P1: the normal force over the phase in which the reference is inside the surface, split into its first 0.5 s after the nominal contact, its middle and its last 0.5 s; the excess above the cap and where it falls; the time from the nominal contact until the force first stays under the cap for 0.2 s; the excess per evaluation payload.", css_class="wide")


def rate_table(caption: str | None = None) -> str:
    """Where the torque rate through P1 comes from (ADR 0011 amendment 3, item 3): the feedforward PD through P1 as modelled, with
    the feedback quantisation removed, with the hold and the bus delay removed, with both removed, with the command quantisation
    removed, and at the torque interface."""
    variants = [("P1 as modelled", RUNS / "metrics_p1.json"), ("exact position and velocity feedback", RUNS / "rate_exact_feedback.json"), ("a frame every millisecond, no bus delay", RUNS / "rate_no_hold.json"),
                ("both", RUNS / "rate_neither.json"), ("exact command channels (gains, torque, goal, velocity)", RUNS / "rate_exact_command.json"), ("P0, the torque interface", RUNS / "metrics_p0.json")]
    rows = []
    for lab, path in variants:
        if not path.exists():
            continue
        _, g = load(path)
        r = g["in_distribution"]["pd_feedforward"]
        rows.append([lab] + [f"{mean_metric(r, reg, 'torque_rate_rms'):.0f}" for reg in ("hold", "free", "policy", "step")] + [f"{1e3 * mean_metric(r, 'free', 'pos_rmse'):.1f}", f"{mean_metric(r, 'blocked', 'excess_force_cap'):.1f}"])
    if len(rows) < 3:
        return ""
    return table(["Through P1 with", "Torque rate, hold (Nm/s)", "free motion", "policy-like", "steps", "Free-motion RMSE (mrad)", "Blocked excess (N)"], rows,
                 caption=caption or "The feedforward PD through P1 (in distribution, 16 cases per family): the torque rate with the sources of the interface's roughness removed one at a time, against the torque interface.", numeric_cols=(1, 2, 3, 4, 5, 6))
