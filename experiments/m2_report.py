"""Build the Milestone 2 report (research/reports/02-known-model) from runs/m2/metrics_test.json and
the training logs. Seeds of one parameterization are pooled per case (mean over seeds) for the
comparison widget; per-seed composites are tabulated separately."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, method_color  # noqa: E402
from report_api import compare_figure, compare_stats, figure, fmt, table  # noqa: E402

from llc.eval.cases import REGIMES  # noqa: E402
from llc.eval.metrics import METRIC_INFO  # noqa: E402

RUN_DIR = Path("runs/m2")
REPORT = Path("research/reports/02-known-model")
BASELINES = [("pd_shared", "PD, shared gains"), ("pd_tuned", "PD, gains tuned per regime"), ("computed_torque", "Computed torque (privileged)")]
KIND_LABELS = {"learned_gains": "Learned PD gains", "pd_residual": "PD + learned residual", "neural_torque": "Neural torque"}
REGIME_LABELS = {"hold": "Hold", "smooth": "Smooth tracking", "zoh": "Policy-like targets (20 Hz hold)", "step": "Step targets", "contact": "Sustained contact", "transition": "Contact transitions"}
PRIMARY = {"hold": ["pos_rmse", "pulse_deflection"], "smooth": ["pos_rmse", "torque_rate_rms"], "zoh": ["pos_rmse", "torque_rate_rms"],
           "step": ["pos_rmse", "settle_time", "torque_rate_rms"], "contact": ["tangential_rmse", "f_std", "contact_breaks"],
           "transition": ["f_peak", "impulse_onset", "release_recovery", "pos_rmse"]}


def pooled_results(data: dict) -> tuple[dict[str, list[dict]], dict[str, list[str]]]:
    """{method: [metrics per case]} with learned runs pooled by kind (mean over seeds per case)."""
    results = {k: v for k, v in data["results"].items() if k in dict(BASELINES)}
    groups: dict[str, list[str]] = {}
    for name, info in data["learned"].items():
        groups.setdefault(info["policy_kind"], []).append(name)
    for kind, names in groups.items():
        pooled = []
        n_cases = len(data["results"][names[0]])
        for i in range(n_cases):
            rows = [data["results"][n][i] for n in names]
            m = {k: float(np.nanmean([r[k] for r in rows])) if isinstance(rows[0][k], float) else rows[0][k] for k in rows[0]}
            m["diverged"] = float(max(r["diverged"] for r in rows))
            pooled.append(m)
        results[kind] = pooled
    return results, groups


def build_compare(results: dict, methods: list[tuple[str, str]], regimes=REGIMES) -> dict:
    metrics_used = [k for k in METRIC_INFO if k != "diverged"]
    values, case_ids = {}, {}
    regimes_ = regimes
    for r in regimes:
        ids = [m["case_id"] for m in results["pd_shared"] if m["regime"] == r]
        case_ids[r] = ids
        values[r] = {}
        for k in metrics_used:
            values[r][k] = {}
            for key, _ in methods:
                rows = {m["case_id"]: m for m in results[key] if m["regime"] == r}
                values[r][k][key] = [rows[c][k] if c in rows and np.isfinite(rows[c][k]) else None for c in ids]
    present = lambda k: any(v is not None for r in regimes_ for vals in values[r][k].values() for v in vals)
    metrics = [{"key": k, "label": METRIC_INFO[k][0], "unit": METRIC_INFO[k][1], "lower_is_better": k != "contact_fraction"} for k in metrics_used if present(k)]
    for r in regimes:
        values[r] = {k: v for k, v in values[r].items() if present(k)}
    return {"methods": [{"key": k, "label": l} for k, l in methods], "regimes": [{"key": r, "label": REGIME_LABELS[r]} for r in regimes],
            "metrics": metrics, "reference_method": "pd_shared", "values": values, "case_ids": case_ids}


def primary_table(cmp: dict, methods: list[tuple[str, str]], caption: str) -> str:
    rows = []
    for r in cmp["values"]:
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
                    sig = "" if s["diff_ci"][0] <= 0.0 <= s["diff_ci"][1] else "*"
                    row.append(f"{fmt(s['mean'], 3)} ({fmt(s['diff'], 2)}{sig})")
            rows.append(row)
    return table(["Regime", "Metric"] + [l for _, l in methods], rows, caption=caption, numeric_cols=tuple(range(2, 2 + len(methods))), css_class="wide")


def training_curves(groups: dict[str, list[str]]) -> str:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for kind, names in groups.items():
        for i, name in enumerate(names):
            recs = [json.loads(l) for l in open(RUN_DIR / name / "log.jsonl") if '"iter"' in l]
            it = [r["iter"] for r in recs]
            axes[0].plot(it, [r["loss"] for r in recs], color=method_color(kind), lw=0.8, alpha=0.9 if i == 0 else 0.5, label=KIND_LABELS[kind] if i == 0 else None)
            vals = [(r["iter"], r["val"]["mean"]) for r in recs if "val" in r]
            if vals:
                axes[1].plot(*zip(*vals), "o-", color=method_color(kind), ms=3, lw=0.8, alpha=0.9 if i == 0 else 0.5, label=KIND_LABELS[kind] if i == 0 else None)
    axes[0].set_yscale("log"); axes[0].set_xlabel("iteration"); axes[0].set_ylabel("training loss (model)")
    axes[1].set_yscale("log"); axes[1].set_xlabel("iteration"); axes[1].set_ylabel("validation composite (MuJoCo, mean over regimes)")
    axes[0].legend(frameon=False, fontsize=8)
    for ax in axes:
        for x in (80, 160):
            ax.axvline(x, color="#cccccc", lw=0.6)
    fig.tight_layout()
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT / "figures" / "training.svg")
    plt.close(fig)
    return figure("figures/training.svg", "Training loss in the differentiable model and the validation composite measured in MuJoCo (lower is better; vertical lines mark curriculum stage changes). One line per seed.", "fig-training", wide=True)


def seed_table(data: dict, groups: dict) -> str:
    rows = []
    for kind, names in groups.items():
        for name in names:
            res = data["results"][name]
            per = [np.mean([m["composite"] for m in res if m["regime"] == r]) for r in REGIMES]
            rows.append([KIND_LABELS[kind], str(data["learned"][name]["seed"])] + [f"{v:.2f}" for v in per] + [str(int(sum(m["diverged"] for m in res)))])
    for key, label in BASELINES:
        res = data["results"][key]
        rows.append([label, "—"] + [f"{np.mean([m['composite'] for m in res if m['regime'] == r]):.2f}" for r in REGIMES] + [str(int(sum(m["diverged"] for m in res)))])
    return table(["Controller", "Seed"] + [REGIME_LABELS[r] for r in REGIMES] + ["Diverged cases"], rows,
                 caption="Composite tuning objective per regime (mean over 16 test cases; lower is better) for every seed, and the number of diverged cases.", numeric_cols=tuple(range(2, 9)))


def gap_table(data: dict, groups: dict) -> str:
    if not data.get("in_model"):
        return ""
    rows = []
    for kind, names in groups.items():
        for name in names:
            if name not in data["in_model"]:
                continue
            for r in REGIMES:
                plant = np.mean([m["composite"] for m in data["results"][name] if m["regime"] == r])
                model = np.mean([m["composite"] for m in data["in_model"][name] if m["regime"] == r])
                p_pos = np.mean([m["pos_rmse"] for m in data["results"][name] if m["regime"] == r])
                m_pos = np.mean([m["pos_rmse"] for m in data["in_model"][name] if m["regime"] == r])
                rows.append([KIND_LABELS[kind], str(data["learned"][name]["seed"]), REGIME_LABELS[r], f"{model:.2f}", f"{plant:.2f}", fmt(m_pos), fmt(p_pos)])
    return table(["Controller", "Seed", "Regime", "Composite in model", "Composite in MuJoCo", "Position RMSE in model (rad)", "Position RMSE in MuJoCo (rad)"], rows,
                 caption="Model-versus-plant gap: the same controller evaluated inside the torch model it was trained in and in MuJoCo, on the same test cases.", numeric_cols=(3, 4, 5, 6))


def ood_table(data: dict, groups: dict) -> str:
    if not data.get("ood"):
        return ""
    methods = [("pd_shared", "PD, shared gains")] + [(k, KIND_LABELS[k]) for k in groups]
    rows = []
    for scale, res_all in data["ood"].items():
        pooled = {k: v for k, v in res_all.items() if k == "pd_shared"}
        for kind, names in groups.items():
            n = len(res_all[names[0]])
            pooled[kind] = [{**res_all[names[0]][i], "composite": float(np.mean([res_all[nm][i]["composite"] for nm in names])),
                            "pos_rmse": float(np.mean([res_all[nm][i]["pos_rmse"] for nm in names])), "diverged": float(max(res_all[nm][i]["diverged"] for nm in names))} for i in range(n)]
        for r in ("smooth", "zoh", "step", "contact", "transition"):
            row = [f"x{scale}", REGIME_LABELS[r]]
            for key, _ in methods:
                rr = [m for m in pooled[key] if m["regime"] == r]
                row.append(f"{np.mean([m['pos_rmse'] for m in rr]):.4f} / {np.mean([m['composite'] for m in rr]):.2f} / {int(sum(m['diverged'] for m in rr))}")
            rows.append(row)
    return table(["Speed scale", "Regime"] + [l for _, l in methods], rows,
                 caption="Out-of-distribution speeds (reference speeds scaled beyond the training range): position RMSE (rad) / composite / diverged cases, 8 cases per regime.", numeric_cols=tuple(range(2, 2 + len(methods))))


def v2_section(narrative: dict[str, str], metrics_path: Path, run_dir: Path) -> str:
    """Protocol v2 results (training with disturbance pulses), appended when available."""
    if not metrics_path.exists():
        return ""
    data = json.load(open(metrics_path))
    results, groups = pooled_results(data)
    methods = BASELINES + [(k, KIND_LABELS[k]) for k in groups]
    cmp = build_compare(results, methods)
    json.dump(cmp, open(REPORT / "data" / "compare_v2.json", "w"))
    global RUN_DIR
    saved, RUN_DIR = RUN_DIR, run_dir
    curves = training_curves(groups).replace("figures/training.svg", "figures/training_v2.svg")
    (REPORT / "figures" / "training_v2.svg").write_bytes((REPORT / "figures" / "training.svg").read_bytes())
    RUN_DIR = saved
    return ("<h2>Protocol v2: disturbance pulses in training</h2>\n" + narrative.get("v2", "") + curves + seed_table(data, groups) +
            primary_table(cmp, methods, "Protocol v2 controllers: primary metrics on the test split (mean over 16 cases); in parentheses the paired difference against PD with shared gains, starred when its 95% bootstrap interval excludes zero.") +
            compare_figure("data/compare_v2.json", cmp, "Per-case metric values on the test split for the protocol v2 controllers.", "fig-compare-v2", default_regime="hold", default_metric="pulse_deflection") +
            narrative.get("v2_findings", ""))


def main(narrative: dict[str, str], metrics_path: Path = RUN_DIR / "metrics_test.json") -> None:
    data = json.load(open(metrics_path))
    results, groups = pooled_results(data)
    methods = BASELINES + [(k, KIND_LABELS[k]) for k in groups]
    (REPORT / "data").mkdir(parents=True, exist_ok=True)
    cmp = build_compare(results, methods)
    json.dump(cmp, open(REPORT / "data" / "compare.json", "w"))
    parts = [f"<h2>Question</h2>\n{narrative['question']}", f"<h2>Setup</h2>\n{narrative['setup']}",
             "<h2>Training</h2>\n" + narrative["training"] + training_curves(groups),
             "<h2>Results in the plant</h2>\n" + narrative["results_intro"] + seed_table(data, groups) +
             primary_table(cmp, methods, "Primary metrics on the test split (mean over 16 cases, seeds pooled per case); in parentheses the paired difference against PD with shared gains, starred when its 95% bootstrap interval excludes zero. Intervals are in the interactive figure below.") +
             compare_figure("data/compare.json", cmp, "Per-case metric values on the test split (16 cases per regime; learned controllers pooled over seeds per case). Dots are cases, the tick the mean, the bar its 95% bootstrap interval.", "fig-compare", default_regime="step", default_metric="pos_rmse"),
             "<h2>Model versus plant</h2>\n" + narrative["gap"] + gap_table(data, groups),
             "<h2>Out-of-distribution speeds</h2>\n" + narrative["ood"] + ood_table(data, groups),
             f"<h2>Findings</h2>\n{narrative['findings']}", f"<h2>Caveats and failures</h2>\n{narrative['caveats']}",
             v2_section(narrative, Path("runs/m2v2/metrics_test.json"), Path("runs/m2v2")),
             f"<h2>Next steps</h2>\n{narrative['next']}"]
    (REPORT / "report.html").write_text("\n\n".join(parts))
    print("wrote", REPORT / "report.html")


if __name__ == "__main__":
    from experiments.m2_narrative import NARRATIVE

    main(NARRATIVE)
