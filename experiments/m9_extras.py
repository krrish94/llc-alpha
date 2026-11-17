"""Milestone 9 report extras: the light-contact detachment sweep (sphere against pads), the task
tables and the plant comparison figures. HTML snippets used by m9_narrative."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, reference_color  # noqa: E402
from report_api import figure, table  # noqa: E402

from experiments.d7_extras import color, label, load, mean_metric  # noqa: E402

REPORT = Path("research/reports/14-m9-plant-v3")
FIG = REPORT / "figures"


def fig_detachment(paths: dict[str, Path], fig_id: str = "fig-detachment") -> str:
    """Contact losses per second of slide against the pressing force, per plant, speed and payload."""
    apply_style()
    data = {name: json.loads(p.read_text()) for name, p in paths.items() if p.exists()}
    if not data:
        return ""
    fig, axes = plt.subplots(1, len(data), figsize=(5.0 * len(data), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, d) in zip(axes, data.items()):
        cells = [c for c in d["cells"] if c["controller"].startswith("F")]
        speeds = sorted({c["speed_scale"] for c in cells}); payloads = sorted({c["payload"] for c in cells})
        styles = {(s, pl): (ls, col) for (s, ls) in zip(speeds, ("-", "--", ":")) for (pl, col) in zip(payloads, ("#2a78d6", "#c2185b", "#6b6b6b"))}
        for s in speeds:
            for pl in payloads:
                rows = sorted([c for c in cells if c["speed_scale"] == s and c["payload"] == pl], key=lambda c: c["f_mean"])
                ls, col = styles[(s, pl)]
                ax.plot([c["f_mean"] for c in rows], [c["contact_loss_rate"] for c in rows], ls, marker="o", ms=3.5, lw=1.0, color=col, label=f"speed x{s:g}, payload {pl:g} kg")
        pd_rows = [c for c in d["cells"] if c["controller"] == "pd"]
        if pd_rows:
            ax.scatter([c["f_mean"] for c in pd_rows], [c["contact_loss_rate"] for c in pd_rows], marker="x", color=reference_color(), label="shared PD (own force)", zorder=3)
        ax.set_title(f"plant {d['plant']}: {'sphere tool' if d['plant'] == 'v2' else 'gripper pads'}", fontsize=10)
        ax.set_xlabel("mean normal force during the slide (N)")
    axes[0].set_ylabel("contact losses per second of slide")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "detachment.svg", bbox_inches="tight"); plt.close(fig)
    return figure("figures/detachment.svg", "Light-contact detachment: constant-force presses (the oracle law with a fixed force around the shared PD) on four table cases of the contact family, contact losses of 5 ms or more per second of slide against the pressing force, for three slide speeds and two payloads, on the sphere tool (plant v2) and on the gripper pads (plant v3). Crosses: the shared PD at its own force.", fig_id, wide=True)


def task_table(groups: dict, methods: list[str], caption: str) -> str:
    rows = []
    for mk in methods:
        if mk not in groups.get("tasks", {}):
            continue
        res = groups["tasks"][mk]
        rows.append([label(mk), f"{100 * mean_metric(res, 'push', 'task_success'):.0f}", f"{100 * mean_metric(res, 'push', 'push_error'):.1f}", f"{100 * mean_metric(res, 'push', 'push_lateral'):.1f}", f"{np.degrees(mean_metric(res, 'push', 'push_yaw')):.0f}",
                     f"{mean_metric(res, 'push', 'object_force_peak_20ms'):.1f}", f"{100 * mean_metric(res, 'pick', 'task_success'):.0f}", f"{100 * mean_metric(res, 'pick', 'place_error'):.1f}", f"{np.degrees(mean_metric(res, 'pick', 'place_tilt')):.0f}",
                     f"{mean_metric(res, 'pick', 'place_force_peak_20ms'):.0f}", f"{1e3 * mean_metric(res, 'pick', 'lift_rmse'):.1f}"])
    return table(["Controller", "Push success (%)", "Push error (cm)", "Lateral (cm)", "Yaw (deg)", "Force on the box, 20 ms peak (N)", "Place success (%)", "Place error (cm)", "Tilt (deg)", "Block-table force, 20 ms peak (N)", "Lift tracking RMSE (mrad)"],
                 rows, caption=caption, numeric_cols=tuple(range(1, 11)))
