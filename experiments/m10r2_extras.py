"""Milestone 10, round 2 report extras: the free-motion figure (tracking error of every method across the
held-out groups, next to the contact composites it must not give up) and the press-band table (the 7 N
floor of the teacher, item C)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, method_color  # noqa: E402
from report_api import figure, table  # noqa: E402

from experiments.d7_extras import label, mean_metric  # noqa: E402
from experiments.d_report import GROUP_LABELS, REGIME_SHORT, add_best_classical  # noqa: E402

REPORT = Path("research/reports/17-m10-round-2")
FIG = REPORT / "figures"
FREE_GROUPS = ("in_distribution", "heldout_workspace", "speed_x1.5", "speed_x2", "payload_3kg", "ood_instances")
SHORT_GROUP = {"in_distribution": "in-distribution", "heldout_workspace": "workspace", "speed_x1.5": "speed x1.5", "speed_x2": "speed x2", "payload_3kg": "payload 3 kg", "ood_instances": "instances"}


def free_motion_figure(groups: dict, methods: list[str], fig_id: str) -> str:
    """Left: free-motion position RMSE of each method per group. Right: the in-distribution cap composites of the
    three contact families, the price (or not) of the free-motion gain."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.6, 1.0]})
    present = [k for k in methods if k in groups["in_distribution"]]
    x = np.arange(len(FREE_GROUPS))
    width = 0.8 / max(1, len(present))
    for i, k in enumerate(present):
        vals = [1e3 * mean_metric(groups[g][k], "free", "pos_rmse") if g in groups and k in groups[g] else np.nan for g in FREE_GROUPS]
        axes[0].bar(x + (i - (len(present) - 1) / 2) * width, vals, width, color=method_color(k), label=label(k))
    axes[0].set_xticks(x); axes[0].set_xticklabels([SHORT_GROUP[g] for g in FREE_GROUPS])
    axes[0].set_ylabel("free-motion position RMSE (mrad)"); axes[0].set_title("Free joint-space motion, per group")
    contact = ("contact2", "blocked", "task")
    x2 = np.arange(len(contact))
    for i, k in enumerate(present):
        vals = [mean_metric(groups["in_distribution"][k], r, "composite_cap") for r in contact]
        axes[1].bar(x2 + (i - (len(present) - 1) / 2) * width, vals, width, color=method_color(k))
    axes[1].set_xticks(x2); axes[1].set_xticklabels([REGIME_SHORT.get(r, r) for r in contact])
    axes[1].set_ylabel("cap composite (lower is better)"); axes[1].set_title("Contact families, in distribution")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "free_motion.svg"); plt.close(fig)
    return figure("figures/free_motion.svg", "Left: free-motion position RMSE of each controller on the in-distribution test cases and on the held-out groups (16 cases each; 3 kg payload and instance deviations change the plant, the speed groups the reference). Right: the cap composites of the three contact families in distribution, which the free-motion gain must not cost.", fig_id, wide=True)


def press_band_table(groups: dict, methods: list[str], caption: str | None = None) -> str:
    """Press-and-slide and the task family: mean normal force while the reference presses, force below the 7 N floor
    (press deficit), force above the 15 N cap, contact losses per second of sliding."""
    rows = []
    res = groups["in_distribution"]
    add_best_classical(res)
    for k in methods:
        if k not in res:
            continue
        row = [label(k)]
        for r in ("contact2", "task"):
            row += [f"{mean_metric(res[k], r, 'f_mean'):.1f}", f"{mean_metric(res[k], r, 'press_deficit'):.2f}", f"{mean_metric(res[k], r, 'excess_force_cap'):.1f}", f"{mean_metric(res[k], r, 'contact_loss_rate'):.1f}"]
        rows.append(row)
    cols = ["mean force (N)", "below 7 N (N)", "above 15 N (N)", "losses (1/s)"]
    return table(["Controller"] + [f"Press-and-slide: {c}" for c in cols] + [f"Task: {c}" for c in cols], rows,
                 caption=caption or "The press band of protocol v3 (7 to 12 N): mean normal force while the reference is inside the surface, mean force below the 7 N floor and above the 15 N cap, and contact losses per second of sliding, on the 16 in-distribution test cases of the press-and-slide and task families.", numeric_cols=tuple(range(1, 9)), css_class="wide")
