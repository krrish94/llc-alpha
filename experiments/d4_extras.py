"""D3 report extras: training curves of the gradient-plan runs (average step reward in the training
medium, plant validation composite and validation excess force). Writes the figure into the report
directory and returns the HTML snippet consumed by experiments/d4_narrative.py."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, method_color  # noqa: E402
from report_api import figure  # noqa: E402

REPORT = Path("research/reports/09-dd-gradients")
LABELS = {"pd_residual_ac": "PD + residual, AC, FD gradients", "neural_torque_ac": "Neural torque, AC, FD gradients",
          "pd_residual_acmed": "PD + residual, AC, median clipping", "neural_torque_acmed": "Neural torque, AC, median clipping",
          "pd_residual_acdec": "PD + residual, AC, learned-model gradients", "neural_torque_acdec": "Neural torque, AC, learned-model gradients",
          "pd_residual_acmed_s02": "PD + residual, AC, median clipping, 20 mrad kernel", "neural_torque_acmed_s02": "Neural torque, AC, median clipping, 20 mrad kernel",
          "pd_residual_acmed_lowrate": "PD + residual, AC, median clipping, rate weight 0.003", "pd_residual_acmed_s02_lowrate": "PD + residual, AC, median clipping, rate weight 0.003, 20 mrad kernel",
          "pd_residual_acmed_lowrate_ref": "PD + residual, AC, median clipping, rate weight 0.003, target history", "neural_torque_acmed_lowrate": "Neural torque, AC, median clipping, rate weight 0.003",
          "pd_residual_acmed_lowrate_imag": "PD + residual, AC, imagined gradients (learned model), rate weight 0.003",
          "neural_torque_acdec_lowrate": "Neural torque, AC, learned-model gradients, rate weight 0.003",
          "pd_residual_acmed_s02_lowrate_ref": "PD + residual, AC, median clipping, rate weight 0.003, 20 mrad kernel, target history",
          "pd_residual_acmed_s02_lowrate_force4": "PD + residual, AC, 20 mrad kernel, force term x4", "pd_residual_acmed_s02_lowrate_force4_nolin": "PD + residual, AC, 20 mrad kernel, force term x4, no restoring term"}
STYLE = {"ac": ":", "acmed": "-", "acdec": "--", "acmed_s02": "-.", "acmed_lowrate": "-", "acmed_s02_lowrate": "-.", "acmed_lowrate_ref": "-", "acmed_lowrate_imag": "--", "acdec_lowrate": "--", "acmed_s02_lowrate_ref": "-.", "acmed_s02_lowrate_force4": ":", "acmed_s02_lowrate_force4_nolin": ":"}


def run_key(name: str) -> str:
    return name.split("-s")[0]


LINE = re.compile(r"\] it +(\d+) st \d+ loss (\S+) reward=(\S+).*?(?:\| val (\S+) exc (\S+))?$")


def read_log(run: Path) -> list[dict]:
    """Training records from log.jsonl or, for runs whose directory was removed, from the text log next to it."""
    if (run / "log.jsonl").exists():
        return [json.loads(l) for l in open(run / "log.jsonl") if '"iter"' in l]
    text = run.with_suffix(".log")
    if not text.exists():
        return []
    recs = []
    for line in open(text):
        m = LINE.search(line.rstrip())
        if not m:
            continue
        rec = {"iter": int(m.group(1)), "loss": float(m.group(2)), "terms": {"reward": float(m.group(3))}}
        if m.group(4):
            rec["val"] = {"mean": float(m.group(4)), "excess_force": float(m.group(5))}
        recs.append(rec)
    return recs


def training_curves(runs: list[str], stage_marks=(100, 200)) -> str:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    seen = set()
    for run in runs:
        recs = read_log(Path(run))
        if not recs:
            continue
        key = run_key(Path(run).name)
        label = LABELS.get(key, key) if key not in seen else None
        seen.add(key)
        ls = STYLE.get(key.split("_", 2)[-1], "-")
        axes[0].plot([r["iter"] for r in recs], [r["terms"].get("reward", -r["loss"]) for r in recs], ls, color=method_color(key), lw=0.8, alpha=0.8, label=label)
        vals = [(r["iter"], r["val"]["mean"], r["val"]["excess_force"]) for r in recs if "val" in r]
        if vals:
            axes[1].plot([v[0] for v in vals], [v[1] for v in vals], ls, marker="o", ms=3, color=method_color(key), lw=0.8, label=label)
            axes[2].plot([v[0] for v in vals], [v[2] for v in vals], ls, marker="o", ms=3, color=method_color(key), lw=0.8, label=label)
    ref = Path("runs/d1/val_pd_shared.json")
    if ref.exists():
        pd = json.loads(ref.read_text())
        axes[1].axhline(pd["mean"], color="#8c8c8c", lw=0.8, ls=(0, (4, 2)), label="PD, shared gains (v2), same cases")
        axes[2].axhline(pd["excess_force"], color="#8c8c8c", lw=0.8, ls=(0, (4, 2)))
    axes[0].set_ylabel("mean step reward, training medium (×100)"); axes[0].set_ylim(-30, 25); axes[1].set_ylabel("validation composite v2 (plant)"); axes[2].set_ylabel("validation excess force (N)")
    for ax in axes:
        ax.set_xlabel("iteration")
        for x in stage_marks:
            ax.axvline(x, color="#cccccc", lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    handles += [h for h, l in zip(h2, l2) if l.startswith("PD, shared")]; labels += [l for l in l2 if l.startswith("PD, shared")]
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT / "figures" / "training.svg", bbox_inches="tight")
    plt.close(fig)
    return figure("figures/training.svg", "Mean step reward in the training medium (bounded kernels minus penalties, summed over a 2 s window and scaled by 1/100), validation composite and validation excess force in the plant (14 fixed cases every 50 iterations). The gray dashed line is the protocol v2 PD on the same validation cases in the corrected plant; the validation of the three first-reward arms and of the collapsing finite-difference arms ran on the nominal robot (erratum, ADR 0005) and is not comparable to it, the rate-light arms' validation is. Vertical lines mark curriculum stage changes; the reward axis is cut at −30 (the collapsing arms fall to −120). Dotted: finite-difference gradients with fixed clipping; solid: per-sample median clipping; dashed: learned-model gradients; dash-dotted: median clipping with the 20 mrad kernel.", "fig-training", wide=True)


if __name__ == "__main__":
    runs = {str(p.with_suffix("")) for d in ("runs/d3", "runs/d4") for p in Path(d).glob("*-s*.log") if not p.name.startswith("killed")}
    print(training_curves(sorted(runs)))
