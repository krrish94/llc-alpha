"""Build the Milestone 1 report (research/reports/01-baselines) from runs/m1/metrics_test.json.

The narrative is written here after inspecting the results, so the whole report is
reproducible from the stored metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, method_color, reference_color  # noqa: E402
from report_api import compare_figure, compare_stats, figure, fmt, table, timeseries_figure  # noqa: E402

from llc.eval.cases import REGIMES  # noqa: E402
from llc.eval.metrics import METRIC_INFO  # noqa: E402

RUN_DIR = Path("runs/m1")
REPORT = Path("research/reports/01-baselines")
METHODS = [("pd_shared", "PD, shared gains"), ("pd_tuned", "PD, gains tuned per regime"), ("pd_initial", "PD, initial hand-set gains"), ("computed_torque", "Computed torque (privileged)")]
REGIME_LABELS = {"hold": "Hold", "smooth": "Smooth tracking", "zoh": "Policy-like targets (20 Hz hold)", "step": "Step targets", "contact": "Sustained contact", "transition": "Contact transitions"}
PRIMARY = {"hold": ["pos_rmse", "pulse_deflection", "pulse_recovery"], "smooth": ["pos_rmse", "vel_rmse", "torque_rate_rms"],
           "zoh": ["pos_rmse", "torque_rate_rms"], "step": ["pos_rmse", "settle_time", "overshoot", "torque_rate_rms"],
           "contact": ["tangential_rmse", "f_std", "contact_breaks", "f_mean"], "transition": ["f_peak", "impulse_onset", "bounces", "release_recovery", "pos_rmse"]}
HIGHER_BETTER = {"contact_fraction"}


def build_compare(data: dict) -> dict:
    results = data["results"]
    metrics_used = [k for k in METRIC_INFO if k not in ("diverged",)]
    values = {}
    case_ids = {}
    regimes_ = REGIMES
    for r in REGIMES:
        values[r] = {}
        ids = [m["case_id"] for m in results["pd_shared"] if m["regime"] == r]
        case_ids[r] = ids
        for k in metrics_used:
            values[r][k] = {}
            for key, _ in METHODS:
                rows = {m["case_id"]: m for m in results[key] if m["regime"] == r}
                values[r][k][key] = [rows[c][k] if c in rows and np.isfinite(rows[c][k]) else None for c in ids]
    present = lambda k: any(v is not None for r in regimes_ for vals in values[r][k].values() for v in vals)
    metrics = [{"key": k, "label": METRIC_INFO[k][0], "unit": METRIC_INFO[k][1], "lower_is_better": k not in HIGHER_BETTER} for k in metrics_used if present(k)]
    for r in REGIMES:
        values[r] = {k: v for k, v in values[r].items() if present(k)}
    return {"methods": [{"key": k, "label": l} for k, l in METHODS], "regimes": [{"key": r, "label": REGIME_LABELS[r]} for r in REGIMES],
            "metrics": metrics, "reference_method": "pd_shared", "values": values, "case_ids": case_ids}


def timeseries(data: dict, regime: str, case_id: str) -> tuple[dict, str]:
    """Time-series JSON and a static SVG for the representative case of a regime."""
    eps = {}
    for key, _ in METHODS:
        p = RUN_DIR / "episodes" / f"{case_id}__{key}.npz"
        if p.exists():
            eps[key] = dict(np.load(p))
    ref_ep = next(iter(eps.values()))
    stride = 5  # 200 Hz in the JSON
    t = ref_ep["t"][::stride]
    series = [{"key": "pos_err", "label": "Max joint tracking error", "unit": "rad"}, {"key": "tau2", "label": "Joint 2 torque", "unit": "Nm"},
              {"key": "tool_z", "label": "Tool height", "unit": "m"}]
    if regime in ("contact", "transition"):
        series.insert(1, {"key": "f_normal", "label": "Normal contact force (full rate)", "unit": "N"})
    values = {s["key"]: {} for s in series}
    for key, ep in eps.items():
        n = len(ep["t"])
        err = np.abs(ep["q_ref"][:n] - ep["q"]).max(1)
        values["pos_err"][key] = err[::stride].round(5).tolist()
        values["tau2"][key] = ep["tau"][::stride, 1].round(3).tolist()
        values["tool_z"][key] = ep["tool_pos"][::stride, 2].round(5).tolist()
        if "f_normal" in values:
            values["f_normal"][key] = ep["f_normal"][::stride].round(2).tolist()
    events = []
    ts = {"t": t.round(4).tolist(), "series": series, "methods": [{"key": k, "label": l} for k, l in METHODS if k in eps], "values": values, "events": events}
    # static figure
    apply_style()
    fig, axes = plt.subplots(len(series), 1, figsize=(7.5, 1.9 * len(series)), sharex=True)
    for ax, s in zip(axes, series):
        for key, label in METHODS:
            if key in values[s["key"]]:
                src = eps[key]
                full = {"pos_err": np.abs(src["q_ref"][:len(src["t"])] - src["q"]).max(1), "tau2": src["tau"][:, 1], "tool_z": src["tool_pos"][:, 2], "f_normal": src["f_normal"]}[s["key"]]
                ax.plot(src["t"], full, color=method_color(key), lw=0.9, label=label)
        ax.set_ylabel(f"{s['label']}\n({s['unit']})", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    svg = f"figures/ts_{regime}.svg"
    fig.savefig(REPORT / svg)
    plt.close(fig)
    return ts, svg


def headroom_table(cmp: dict) -> str:
    rows = []
    for r in REGIMES:
        for k in PRIMARY[r]:
            if k not in cmp["values"][r]:
                continue
            st = {s["key"]: s for s in compare_stats(cmp, r, k)}
            sh, tu = st["pd_shared"], st["pd_tuned"]
            rel = f"{100.0 * tu['diff'] / sh['mean']:+.0f}%" if sh["mean"] else "—"
            rows.append([REGIME_LABELS[r], f"{METRIC_INFO[k][0]} ({METRIC_INFO[k][1]})", fmt(sh["mean"]), fmt(tu["mean"]),
                         f"{fmt(tu['diff'])} [{fmt(tu['diff_ci'][0])}, {fmt(tu['diff_ci'][1])}]", rel, str(sh["n"])])
    return table(["Regime", "Metric", "Shared gains", "Per-regime gains", "Paired difference [95% CI]", "Relative", "n"], rows,
                 caption="Headroom of per-regime tuning over shared gains on the primary metrics (test split, paired by case; negative differences favour per-regime tuning for lower-is-better metrics).",
                 numeric_cols=(2, 3, 4, 5, 6))


def gains_table(data: dict) -> str:
    rows = []
    for name, label in [("pd_initial", "initial"), ("pd_shared", "shared")] + [(r, REGIME_LABELS[r]) for r in REGIMES]:
        spec = data["controllers"][name] if name in ("pd_initial", "pd_shared") else data["controllers"]["pd_tuned"][name]
        kp = spec["kp"]
        tuning_key = {"pd_shared": "shared"}.get(name, name)
        zeta = np.exp(data["tuning"][tuning_key]["params"][7]) if tuning_key in data["tuning"] else 1.0
        rows.append([label] + [f"{v:.0f}" for v in kp] + [f"{zeta:.2f}"])
    return table(["Gain set"] + [f"kp{j+1}" for j in range(7)] + ["damping ratio"], rows,
                 caption="PD stiffness per joint (Nm/rad) and the common damping ratio; damping follows kd_i = 2 zeta sqrt(kp_i M_ii) with M_ii the nominal diagonal inertia at the home configuration.", numeric_cols=tuple(range(1, 9)))


def tuning_table(data: dict) -> str:
    rows = [[name, f"{g['x0_objective']:.3f}", f"{g['objective']:.3f}", str(g["episodes_evaluated"]), f"{g['wall_s']/60:.1f}"] for name, g in data["tuning"].items()]
    return table(["Target", "Composite, initial gains", "Composite, tuned", "Episodes evaluated", "Wall time (min)"], rows,
                 caption="Tuning cost and objective on the tune split (6 cases per regime; CMA-ES, population 12, 40 iterations).", numeric_cols=(1, 2, 3, 4))


def main(narrative: dict[str, str]) -> None:
    data = json.load(open(RUN_DIR / "metrics_test.json"))
    (REPORT / "data").mkdir(parents=True, exist_ok=True)
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    cmp = build_compare(data)
    json.dump(cmp, open(REPORT / "data" / "compare.json", "w"))
    parts = [f"<h2>Question</h2>\n{narrative['question']}",
             f"<h2>Setup</h2>\n{narrative['setup']}",
             "<h2>Baselines and tuning</h2>\n" + narrative["baselines"] + gains_table(data) + tuning_table(data),
             "<h2>Results</h2>\n" + narrative["results_intro"] + headroom_table(cmp) +
             compare_figure("data/compare.json", cmp, "Per-case metric values on the test split (16 cases per regime). Dots are cases, the tick is the mean and the bar its 95% bootstrap interval; the table gives the paired difference against the shared gain set. The composite is the tuning objective, not a research signal.", "fig-compare", default_regime="contact", default_metric="f_std")]
    ts_parts = []
    for r in REGIMES:
        cid = data["representative_cases"][r]
        ts, svg = timeseries(data, r, cid)
        json.dump(ts, open(REPORT / "data" / f"ts_{r}.json", "w"))
        ts_parts.append(timeseries_figure(f"data/ts_{r}.json", f"{REGIME_LABELS[r]}: representative case {cid} (median pd_shared composite). Contact force is shown at the native 1 kHz rate in the static figure and at 200 Hz in the interactive view.", f"fig-ts-{r}", svg))
    parts.append("<h2>Representative episodes</h2>\n" + narrative["episodes_intro"] + "\n".join(ts_parts))
    parts.append(f"<h2>Findings</h2>\n{narrative['findings']}")
    parts.append(f"<h2>Caveats and failures</h2>\n{narrative['caveats']}")
    parts.append(f"<h2>Next steps</h2>\n{narrative['next']}")
    (REPORT / "report.html").write_text("\n\n".join(parts))
    print("wrote", REPORT / "report.html")


if __name__ == "__main__":
    from experiments.m1_narrative import NARRATIVE

    main(NARRATIVE)
