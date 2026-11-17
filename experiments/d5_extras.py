"""D4 (PPO) report extras: training curves (return per window, reflex rate, exploration std, validation
composite and excess force in the corrected plant). Returns the HTML snippet used by d5_narrative."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, method_color  # noqa: E402
from report_api import figure  # noqa: E402

REPORT = Path("research/reports/10-dd-ppo")
LABELS = {"ppo_all": "PPO, all families, from the residual (10 ms decisions)", "ppo_contact": "PPO, contact families", "ppo_free": "PPO, free-space families",
          "ppo_all_scratch": "PPO, all families, from scratch", "ppo_all_hold1": "PPO, all families, 1 ms decisions", "ppo_all_hold5": "PPO, all families, 5 ms decisions",
          "ppo_all_filt": "PPO, all families, reflex on the 20 ms filtered force", "ppo_contact_filt": "PPO, contact families, reflex on the 20 ms filtered force",
          "ppo_all_feas": "PPO, all families, feasible-direction kernel in contact", "ppo_contact_feas": "PPO, contact families, feasible-direction kernel in contact",
          "ppo_all_v2": "PPO, all families, non-terminating reflex, penalty from 30 N, feasible kernel", "ppo_contact_v2": "PPO, contact families, non-terminating reflex, penalty from 30 N, feasible kernel",
          "ppo_all_v3": "PPO, all families, as v2 with 4 s windows and a drift penalty to the warm start",
          "ppo_all_full": "PPO, all families plus task, 6 s windows, drift penalty (declared, task no longer held out)",
          "ppo_all_v4": "PPO, all families, as v3 with the normal direction rewarded up to 30 N", "ppo_all_v4full": "PPO, all families plus task, 6 s windows, normal direction rewarded up to 30 N"}


def run_key(name: str) -> str:
    return name.split("-s")[0]


def training_curves(runs: list[str]) -> str:
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    seen = set()
    for run in runs:
        log = Path(run) / "log.jsonl"
        if not log.exists():
            continue
        recs = [json.loads(l) for l in open(log)]
        key = run_key(Path(run).name)
        label = LABELS.get(key, key) if key not in seen else None
        seen.add(key)
        col = method_color(key)
        tr = [r for r in recs if "mean_return" in r]
        axes[0].plot([r["update"] for r in tr], [r["mean_return"] for r in tr], color=col, lw=0.8, label=label)
        axes[1].plot([r["update"] for r in tr], [r["reflex_rate"] for r in tr], color=col, lw=0.8)
        vals = [(r["update"], r["val"]["mean"], r["val"]["excess_force"]) for r in recs if "val" in r]
        if vals:
            axes[2].plot([v[0] for v in vals], [v[1] for v in vals], marker="o", ms=3, color=col, lw=0.8)
            axes[3].plot([v[0] for v in vals], [v[2] for v in vals], marker="o", ms=3, color=col, lw=0.8)
    ref = Path("runs/d1/val_pd_shared.json")
    if ref.exists():
        pd = json.loads(ref.read_text())
        axes[2].axhline(pd["mean"], color="#8c8c8c", lw=0.8, ls=(0, (4, 2)), label="PD, shared gains (v2), same cases")
        axes[3].axhline(pd["excess_force"], color="#8c8c8c", lw=0.8, ls=(0, (4, 2)))
    axes[0].set_ylabel("return per 2 s window"); axes[1].set_ylabel("reflex rate (share of windows above 40 N)"); axes[2].set_ylabel("validation composite v2"); axes[3].set_ylabel("validation excess force (N)")
    for ax in axes:
        ax.set_xlabel("PPO update")
    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT / "figures" / "training.svg", bbox_inches="tight")
    plt.close(fig)
    return figure("figures/training.svg", "PPO training: mean return per 2 s window (reward units of the actor-critic objective), share of windows that triggered the 40 N reflex, validation composite and validation excess force on the 14 fixed cases of the corrected plant (every 20 updates; the point at update −1 is the warm start under the decision period). The gray dashed line is the protocol v2 PD on the same cases.", "fig-training", wide=True)


if __name__ == "__main__":
    print(training_curves(sorted(str(p) for p in Path("runs/d5").glob("*-s*") if p.is_dir())))
