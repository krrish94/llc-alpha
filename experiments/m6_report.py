"""Build the Milestone 6 report (research/reports/06-phases) from runs/m6/phases.json: transition
and sustained-contact metrics over the speed x payload grid for fixed gains, phase-scheduled gains
with three blend times, and the learned controllers."""
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
from report_api import bootstrap_ci, figure, fmt, table  # noqa: E402

from llc.eval.metrics import METRIC_INFO  # noqa: E402

RUN_DIR = Path("runs/m6")
REPORT = Path("research/reports/06-phases")
METRICS = {"transition": ["f_peak", "impulse_onset", "bounces", "f_mean", "release_recovery", "torque_rate_max", "pos_rmse"],
           "contact": ["f_peak", "tangential_rmse", "f_std", "contact_breaks", "f_mean", "torque_rate_max", "pos_rmse"]}


def palette_key(name: str) -> str:
    if name.startswith("pd_phase"):
        return "pd_scheduled"
    if name in ("pd_contact_gains", "pd_transition_gains"):
        return "pd_tuned"
    return name.split("-s")[0]


def grid_table(data: dict, labels: dict, regime: str) -> str:
    rows = []
    for key, res in data["grid"].items():
        r, speed, payload = key.split("|")
        if r != regime:
            continue
        for name in data["controllers"]:
            vals = res[name]
            row = [speed.replace("speed", "x"), payload.replace("payload", "") + " kg", labels.get(name, name)]
            for m in METRICS[regime]:
                xs = [v[m] for v in vals if np.isfinite(v[m])]
                row.append(fmt(np.mean(xs), 3) if xs else "—")
            row.append(str(int(sum(v["diverged"] for v in vals))))
            rows.append(row)
    return table(["Speed", "Payload", "Controller"] + [f"{METRIC_INFO[m][0]} ({METRIC_INFO[m][1]})" for m in METRICS[regime]] + ["Diverged"], rows,
                 caption=f"{'Contact transitions' if regime == 'transition' else 'Sustained contact'}: metric means over {data['n_cases']} cases per cell.", numeric_cols=tuple(range(3, 3 + len(METRICS[regime]) + 1)), css_class="wide")


def summary_figure(data: dict, labels: dict) -> str:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    panels = [("transition", "f_peak"), ("transition", "impulse_onset"), ("contact", "tangential_rmse")]
    for ax, (regime, metric) in zip(axes, panels):
        cells = [k for k in data["grid"] if k.startswith(regime)]
        x = np.arange(len(cells))
        names = list(data["controllers"])
        w = 0.8 / len(names)
        for i, name in enumerate(names):
            means, los, his = [], [], []
            for k in cells:
                vals = [v[metric] for v in data["grid"][k][name] if np.isfinite(v[metric])]
                means.append(np.mean(vals) if vals else np.nan); ci = bootstrap_ci(vals, f"{k}/{name}/{metric}") if vals else (np.nan, np.nan); los.append(ci[0]); his.append(ci[1])
            means, los, his = map(np.array, (means, los, his))
            ax.bar(x + (i - len(names) / 2 + 0.5) * w, means, w, color=method_color(palette_key(name)), label=labels.get(name, name), yerr=[means - los, his - means], error_kw={"lw": 0.6})
        ax.set_xticks(x); ax.set_xticklabels([k.split("|", 1)[1].replace("speed", "x").replace("|payload", "\n") + " kg" for k in cells], fontsize=7)
        ax.set_ylabel(f"{METRIC_INFO[metric][0]} ({METRIC_INFO[metric][1]})", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT / "figures" / "phases_summary.svg")
    plt.close(fig)
    return figure("figures/phases_summary.svg", "Peak impact force and onset impulse (transitions) and tangential error during sliding (sustained contact) over the speed x payload grid; bars are means with 95% bootstrap intervals over cases.", "fig-phases", wide=True)


def main(narrative: dict[str, str], labels: dict[str, str]) -> None:
    data = json.load(open(RUN_DIR / "phases.json"))
    parts = [f"<h2>Question</h2>\n{narrative['question']}", f"<h2>Setup</h2>\n{narrative['setup']}",
             "<h2>Results</h2>\n" + narrative["results_intro"] + summary_figure(data, labels) + grid_table(data, labels, "transition") + grid_table(data, labels, "contact"),
             f"<h2>Findings</h2>\n{narrative['findings']}", f"<h2>Caveats and failures</h2>\n{narrative['caveats']}", f"<h2>Next steps</h2>\n{narrative['next']}"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report.html").write_text("\n\n".join(parts))
    print("wrote", REPORT / "report.html")


if __name__ == "__main__":
    from experiments.m6_narrative import LABELS, NARRATIVE

    main(NARRATIVE, LABELS)
