"""Generic report builder for the deep-dive reports (protocol v2 metrics files written by
experiments/d_evaluate.py): per-group tables of the primary metrics, the comparison widget per
group, and the narrative supplied by a module.

    uv run python experiments/d_report.py --metrics runs/d1/metrics_test.json --report 07-dd-baselines --narrative experiments.d1_narrative
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from report_api import compare_figure, compare_stats, fmt, table  # noqa: E402

from llc.eval.cases_v2 import REGIMES_V2  # noqa: E402
from llc.eval.metrics import METRIC_INFO  # noqa: E402

REGIME_LABELS = {"hold": "Hold", "free": "Free joint-space motion", "policy": "Policy-like targets", "step": "Step targets",
                 "contact2": "Contact (tilt, depth, slide)", "blocked": "Blocked reference", "task": "Task sequence", "push": "Box pushing", "pick": "Pick-and-place"}
GROUP_LABELS = {"in_distribution": "In-distribution families", "heldout_workspace": "Held-out workspace (base yaw 1.2 to 2.6 rad)",
                "speed_x1.5": "Speed x1.5", "speed_x2": "Speed x2", "payload_3kg": "Payload 3 kg", "ood_instances": "Held-out robot instances (doubled deviations)",
                "surface_2000": "Compliant surface, 2 kN/m (held out)", "surface_500": "Compliant surface, 500 N/m (held out)", "tasks": "Task families (box pushing, pick-and-place)", "latency_5ms": "In-distribution families with a 5 ms compute latency (learned controllers)"}
REGIME_LABELS.setdefault("push", "Box pushing"); REGIME_LABELS.setdefault("pick", "Pick-and-place"); REGIME_LABELS.setdefault("torque", "Torque tracking")
for _k, _v in (("wipe", "Wiping (raster under the force band)"), ("erase", "Erasing with a held block"), ("peg", "Peg-in-hole with a chamfer"), ("drawer", "Drawer pushed closed"), ("door", "Door pushed closed")):
    REGIME_LABELS.setdefault(_k, _v)
from llc.sim.plant import PLANT  # noqa: E402

GROUP_LABELS.setdefault(f"payload_{PLANT.payload_heavy:g}kg", f"Payload {PLANT.payload_heavy:g} kg (held out)")  # the plant's heavy-payload group (ADR 0011)
GROUP_LABELS.setdefault("torque_tracking", "Torque-tracking family (the interface alone)")
PRIMARY = {"hold": ["pos_rmse", "pulse_deflection", "torque_rate_rms"], "free": ["pos_rmse", "torque_rate_rms"], "policy": ["pos_rmse", "torque_rate_rms"],
           "step": ["pos_rmse", "settle_time", "overshoot"],
           "contact2": ["excess_force", "excess_force_cap", "f_peak", "f_peak_20ms", "feasible_tracking", "press_deficit", "contact_breaks", "contact_loss_rate", "contact_impedance"],
           "blocked": ["excess_force", "excess_force_cap", "f_peak", "f_peak_20ms", "blocked_drift", "blocked_motion", "apparent_stiffness", "contact_impedance", "contact_recovery", "feasible_tracking"],
           "task": ["excess_force", "excess_force_cap", "f_peak", "f_peak_20ms", "feasible_tracking", "press_deficit", "contact_loss_rate"],
           "push": ["task_success", "push_error", "push_lateral", "push_yaw", "object_force_peak_20ms", "excess_force_cap", "torque_rate_rms"],
           "pick": ["task_success", "grasp_success", "pregrasp_disturbance", "grasp_force_peak_20ms", "place_error", "place_tilt", "place_force_peak_20ms", "lift_rmse", "torque_rate_rms"],
           "wipe": ["excess_force_cap", "f_peak_20ms", "press_deficit", "contact_loss_rate", "tangential_rmse", "torque_rate_rms"],
           "erase": ["excess_force_cap", "f_peak_20ms", "press_deficit", "contact_loss_rate", "tangential_rmse", "torque_rate_rms"],
           "peg": ["task_success", "insertion_depth", "depth_deficit", "excess_force_cap", "f_peak_20ms", "torque_rate_rms"],
           "drawer": ["task_success", "travel", "excess_force_cap", "f_peak_20ms", "contact_loss_rate", "torque_rate_rms"],
           "door": ["task_success", "travel", "excess_force_cap", "f_peak_20ms", "contact_loss_rate", "torque_rate_rms"],
           "torque": ["tau_track_rmse", "tau_track_delay", "tau_track_bandwidth", "saturation", "torque_rate_rms"]}


def pooled(results: dict, learned: dict) -> tuple[dict, dict]:
    """Pool seeds per (kind, variant) by averaging metrics per case."""
    groups: dict[str, list[str]] = {}
    for name, info in learned.items():
        key = info.get("method_key") or info["policy_kind"] + (f"_{info['variant']}" if info.get("variant") else "")
        groups.setdefault(key, []).append(name)
    out = {k: v for k, v in results.items() if k not in learned}
    for key, names in groups.items():
        names = [nm for nm in names if nm in results]  # a group evaluated without the learned runs (the torque-tracking family, ADR 0011)
        if not names:
            continue
        n = len(results[names[0]])
        out[key] = []
        for i in range(n):
            rows = [results[nm][i] for nm in names]
            m = {k: (float(np.nanmean([r.get(k, float("nan")) for r in rows])) if isinstance(rows[0][k], float) else rows[0][k]) for k in rows[0]}
            m["diverged"] = float(max(r["diverged"] for r in rows))
            out[key].append(m)
    return out, groups


CLASSICAL_KEYS = ("pd_shared", "pd_shared_cap", "pd_feedforward", "pd_shared_cap_feedforward", "pd_shared_v2gains", "pd_shared_v1", "pd_shipped", "pd_shipped_feedforward", "pd_retuned")  # the shipped gains of a robot's driver and the PD re-tuned through a profile count as classical (ADR 0011); the per-family PD (pd_family, ADR 0012) is a separate reference


def add_best_classical(results: dict, objective: str = "composite_cap", keys=CLASSICAL_KEYS) -> dict:
    """Adds the synthetic method ``best_classical``: per regime, the per-case rows of the classical controller
    with the lowest mean ``objective`` over the regime's cases (ADR 0010: the reference of every comparison)."""
    if "best_classical" in results:  # chosen once; a caller that picked the candidates keeps its choice
        return results
    present = [k for k in keys if k in results]
    if not present:
        return results
    regimes = sorted({m["regime"] for m in results[present[0]]})
    rows = []
    for r in regimes:
        best = min(present, key=lambda k: float(np.nanmean([m[objective] for m in results[k] if m["regime"] == r])))
        rows += [dict(m, best_classical_source=best) for m in results[best] if m["regime"] == r]
    order = {m["case_id"]: i for i, m in enumerate(results[present[0]])}
    results["best_classical"] = sorted(rows, key=lambda m: order.get(m["case_id"], 0))
    return results


def build_compare(results: dict, methods: list[tuple[str, str]], reference: str) -> dict:
    regimes = [r for r in list(REGIMES_V2) + ["push", "pick", "wipe", "erase", "peg", "drawer", "door"] if any(m["regime"] == r for m in results[reference])]
    metrics_used = [k for k in METRIC_INFO if k != "diverged"]
    values, case_ids = {}, {}
    for r in regimes:
        ids = [m["case_id"] for m in results[reference] if m["regime"] == r]
        case_ids[r] = ids
        values[r] = {}
        for k in metrics_used:
            values[r][k] = {}
            for key, _ in methods:
                rows = {m["case_id"]: m for m in results[key] if m["regime"] == r}
                values[r][k][key] = [rows[c][k] if c in rows and np.isfinite(rows[c].get(k, float("nan"))) else None for c in ids]  # metrics added later are absent from older tables
    present = {k for k in metrics_used if any(v is not None for r in regimes for vals in values[r][k].values() for v in vals)}
    metrics = [{"key": k, "label": METRIC_INFO[k][0], "unit": METRIC_INFO[k][1], "lower_is_better": k not in ("contact_fraction", "task_success", "object_contact_fraction")} for k in metrics_used if k in present]
    for r in regimes:
        values[r] = {k: v for k, v in values[r].items() if k in present}
    return {"methods": [{"key": k, "label": l} for k, l in methods], "regimes": [{"key": r, "label": REGIME_LABELS[r]} for r in regimes],
            "metrics": metrics, "reference_method": reference, "values": values, "case_ids": case_ids}


def primary_table(cmp: dict, methods: list[tuple[str, str]], caption: str) -> str:
    rows = []
    for r in [x["key"] for x in cmp["regimes"]]:
        for k in PRIMARY[r]:
            if k not in cmp["values"][r]:
                continue
            st = {s["key"]: s for s in compare_stats(cmp, r, k)}
            row = [REGIME_LABELS[r], f"{METRIC_INFO[k][0]} ({METRIC_INFO[k][1]})"]
            for key, _ in methods:
                s = st.get(key)
                if s is None or not np.isfinite(s["mean"]):
                    row.append("—")
                elif s["is_ref"]:
                    row.append(fmt(s["mean"], 3))
                else:
                    sig = "" if (np.isnan(s["diff_ci"][0]) or s["diff_ci"][0] <= 0.0 <= s["diff_ci"][1]) else "*"
                    row.append(f"{fmt(s['mean'], 3)} ({fmt(s['diff'], 2)}{sig})")
            rows.append(row)
    return table(["Regime", "Metric"] + [l for _, l in methods], rows, caption=caption, numeric_cols=tuple(range(2, 2 + len(methods))), css_class="wide")


def summary_table(data: dict, methods: list[tuple[str, str]], pooled_groups: dict) -> str:
    rows = []
    for gname, results in pooled_groups.items():
        for key, label in methods:
            res = results.get(key)
            if not res:
                continue
            has_cap = all("excess_force_cap" in m for m in res)
            rows.append([GROUP_LABELS.get(gname, gname), label, fmt(float(np.mean([m["composite"] for m in res])), 3),
                         fmt(float(np.mean([m["composite_cap"] for m in res])), 3) if has_cap else "—",
                         fmt(float(np.mean([m["excess_force"] for m in res])), 3), fmt(float(np.mean([m["excess_force_cap"] for m in res])), 3) if has_cap else "—",
                         fmt(float(np.nanmean([m["f_peak"] for m in res])), 3), fmt(float(np.nanmean([m.get("f_peak_20ms", float("nan")) for m in res])), 3) if has_cap else "—",
                         fmt(float(np.mean([m["pos_rmse"] for m in res])), 3), str(int(sum(m["diverged"] for m in res)))])
    from llc.eval.metrics import FORCE_CAP, FORCE_SOFT_LIMIT

    return table(["Group", "Controller", "Composite v2", f"Composite ({FORCE_CAP:g} N cap)", f"Excess force above {FORCE_SOFT_LIMIT:g} N (N)", f"Excess force above {FORCE_CAP:g} N (N)", "Peak force (N)", "Sustained peak, 20 ms (N)", "Position RMSE (rad)", "Diverged"], rows,
                 caption="Means over all cases of a group (every regime of the group pooled). The composites are tuning and selection objectives, not research signals; the cap columns follow ADR 0007.", numeric_cols=(2, 3, 4, 5, 6, 7, 8, 9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--narrative", required=True, help="module exposing NARRATIVE (dict) and METHODS (list of (key, label))")
    ap.add_argument("--reference", default="pd_shared", help="method key of the paired comparisons; 'best_classical' takes the best classical controller per regime (ADR 0010)")
    args = ap.parse_args()
    nar = importlib.import_module(args.narrative)
    data = json.load(open(args.metrics))
    report = Path("research/reports") / args.report
    (report / "data").mkdir(parents=True, exist_ok=True)
    pooled_groups, groups = {}, {}
    for gname, results in data["groups"].items():
        pooled_groups[gname], groups = pooled(results, data["learned"])
        if args.reference == "best_classical":
            add_best_classical(pooled_groups[gname])
    methods = [(k, l) for k, l in nar.METHODS if k in pooled_groups["in_distribution"]]
    parts = [f"<h2>Question</h2>\n{nar.NARRATIVE['question']}"]
    if nar.NARRATIVE.get("erratum"):
        parts.append(f"<h2>Erratum</h2>\n{nar.NARRATIVE['erratum']}")
    parts.append(f"<h2>Setup</h2>\n{nar.NARRATIVE['setup']}")
    for extra in nar.NARRATIVE.get("extra_sections", []):
        parts.append(extra)
    parts.append("<h2>Summary over groups</h2>\n" + nar.NARRATIVE.get("summary_intro", "") + summary_table(data, methods, pooled_groups))
    for gname, results in pooled_groups.items():
        if args.reference not in results:  # a group evaluated without the reference (the torque-tracking family runs the torque program only, ADR 0011) is reported by its own extras
            continue
        cmp = build_compare(results, methods, args.reference)
        json.dump(cmp, open(report / "data" / f"compare_{gname}.json", "w"))
        default_regime = cmp["regimes"][0]["key"]
        default_metric = "excess_force" if default_regime in ("contact2", "blocked", "task") else "pos_rmse"
        parts.append(f"<h2>{GROUP_LABELS.get(gname, gname)}</h2>\n" + nar.NARRATIVE.get(f"group_{gname}", "") +
                     primary_table(cmp, methods, f"{GROUP_LABELS.get(gname, gname)}: primary metrics (mean over cases; paired difference against {'the best classical controller of each family' if args.reference == 'best_classical' else 'PD with shared gains'}, starred when the 95% bootstrap interval excludes zero).") +
                     compare_figure(f"data/compare_{gname}.json", cmp, f"{GROUP_LABELS.get(gname, gname)}: per-case values; dots are cases, the tick the mean, the bar its 95% bootstrap interval.", f"fig-compare-{gname}", default_regime=default_regime, default_metric=default_metric))
    parts += [f"<h2>Findings</h2>\n{nar.NARRATIVE['findings']}", f"<h2>Caveats and failures</h2>\n{nar.NARRATIVE['caveats']}", f"<h2>Next steps</h2>\n{nar.NARRATIVE['next']}"]
    (report / "report.html").write_text("\n\n".join(parts))
    print("wrote", report / "report.html")


if __name__ == "__main__":
    main()


REGIME_SHORT = {"hold": "Hold (10 N pulses)", "free": "Free motion", "policy": "Policy-like targets", "step": "Step targets", "contact2": "Contact (press, slide)", "blocked": "Blocked target", "task": "Task (held out)", "push": "Box pushing", "pick": "Pick-and-place",
                "wipe": "Wiping", "erase": "Erasing (held block)", "peg": "Peg-in-hole", "drawer": "Drawer", "door": "Door"}


def wins_table(data: dict, methods: list[tuple[str, str]], reference: str = "pd_shared", caption: str | None = None) -> str:
    """Where the learned controllers beat or lose to the reference: one row per (group, family, primary
    metric) with a significant paired difference (95% bootstrap CI on per-case differences excludes 0),
    listing each method's mean against the reference's and the relative change; parity rows are left out."""
    from llc.eval.metrics import METRIC_INFO

    rows = []
    higher = ("contact_fraction", "task_success", "object_contact_fraction")
    for gname, results in data["groups"].items():
        pooled_res, _ = pooled(results, data["learned"])
        if reference == "best_classical":
            add_best_classical(pooled_res)
        if reference not in pooled_res:  # a group without the reference (the torque-tracking family runs the torque program only)
            continue
        present = [m for m in methods if m[0] in pooled_res]
        cmp = build_compare(pooled_res, present, reference)
        for r in [x["key"] for x in cmp["regimes"]]:
            for k in PRIMARY[r]:
                if k not in cmp["values"][r]:
                    continue
                st = {s["key"]: s for s in compare_stats(cmp, r, k)}
                ref = st[reference]["mean"]
                better, worse = [], []
                for key, label in present:
                    s = st.get(key)
                    if key == reference or s is None or not np.isfinite(s["mean"]):
                        continue
                    lo, hi = s["diff_ci"]
                    if np.isnan(lo) or lo <= 0.0 <= hi:
                        continue
                    rel = (s["mean"] - ref) / (abs(ref) + 1e-9) * 100
                    txt = f"{label}: {fmt(s['mean'], 3)} ({rel:+.0f}%)"
                    (better if (s["diff"] < 0) == (k not in higher) else worse).append(txt)
                if better or worse:
                    rows.append([GROUP_LABELS.get(gname, gname), REGIME_SHORT.get(r, r), f"{METRIC_INFO[k][0]} ({METRIC_INFO[k][1]})", fmt(ref, 3), "; ".join(better) or "—", "; ".join(worse) or "—"])
    return table(["Group", "Family", "Metric", "Reference", "Significantly better", "Significantly worse"], rows,
                 caption=caption or "Scenarios and metrics with a significant paired difference against the reference (95% bootstrap CI on per-case differences; 16 cases per family and group). Relative change in parentheses; rows at parity are omitted.",
                 numeric_cols=(3,), css_class="wide")
