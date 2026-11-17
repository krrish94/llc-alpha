"""D2 report extras: training curves of the deep-dive runs (loss in the simulator medium and the
plant validation composite) and the mode-detection probe table. Writes figures into the report
directory and returns HTML snippets consumed by experiments/d2_narrative.py."""
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
from report_api import figure, fmt, table  # noqa: E402

REPORT = Path("research/reports/08-dd-controllers")
LABELS = {"pd_residual": "PD + residual", "neural_torque": "Neural torque", "pd_residual_noforce": "PD + residual, no force terms", "neural_torque_noforce": "Neural torque, no force terms"}


def run_key(name: str) -> str:
    return name.split("-s")[0]


def training_curves(runs: list[str]) -> str:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    seen = set()
    for run in runs:
        recs = [json.loads(l) for l in open(Path(run) / "log.jsonl") if '"iter"' in l]
        key = run_key(Path(run).name)
        label = LABELS.get(key, key) if key not in seen else None
        seen.add(key)
        ls = "--" if "noforce" in key else "-"
        axes[0].plot([r["iter"] for r in recs], [r["loss"] for r in recs], ls, color=method_color(key), lw=0.8, alpha=0.8, label=label)
        vals = [(r["iter"], r["val"]["mean"], r["val"]["excess_force"]) for r in recs if "val" in r]
        if vals:
            axes[1].plot([v[0] for v in vals], [v[1] for v in vals], ls, marker="o", ms=3, color=method_color(key), lw=0.8, label=label)
            axes[2].plot([v[0] for v in vals], [v[2] for v in vals], ls, marker="o", ms=3, color=method_color(key), lw=0.8, label=label)
    axes[0].set_yscale("log"); axes[0].set_ylabel("training loss (simulator)"); axes[1].set_yscale("log"); axes[1].set_ylabel("validation composite v2 (plant)"); axes[2].set_ylabel("validation excess force (N)")
    for ax in axes:
        ax.set_xlabel("iteration")
        for x in (100, 200):
            ax.axvline(x, color="#cccccc", lw=0.6)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT / "figures" / "training.svg")
    plt.close(fig)
    return figure("figures/training.svg", "Training loss in the simulator medium, validation composite and validation excess force in the plant (14 fixed cases every 50 iterations). Vertical lines mark curriculum stage changes; dashed lines are the no-force ablations.", "fig-training", wide=True)


def probe_table(probe_files: dict[str, str]) -> str:
    rows = []
    for name, path in probe_files.items():
        if not Path(path).exists():
            continue
        p = json.load(open(path))
        rows.append([LABELS.get(run_key(name), name), fmt(p["hidden_probe"]["balanced_accuracy"], 3), fmt(p["raw_probe"]["balanced_accuracy"], 3),
                     fmt(p["onset_lead_ms_hidden"]["median"], 3) if p["onset_lead_ms_hidden"]["median"] is not None else "—",
                     fmt(p["onset_lead_ms_raw"]["median"], 3) if p["onset_lead_ms_raw"]["median"] is not None else "—", str(p["n_samples_test"]), fmt(p["contact_fraction"], 2)])
    if not rows:
        return ""
    return table(["Controller", "Balanced accuracy, hidden layer", "Balanced accuracy, raw observation", "Onset detection lead (ms), hidden", "Onset detection lead (ms), raw", "Test samples", "Contact fraction"], rows,
                 caption="Linear probe for the contact state: logistic regression on the last hidden layer and, as reference, on the raw observation, trained on half of 24 validation episodes (contact2, blocked, free, task) and tested on the other half; lead is the median time between the probe's first positive and the first measured contact (negative = before contact).", numeric_cols=(1, 2, 3, 4, 5, 6))
