"""Shared matplotlib style for notebook plots. Colors come from research/design.json."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
from cycler import cycler

DESIGN_PATH = Path(__file__).with_name("design.json")
FONT_FAMILY = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


@lru_cache(maxsize=1)
def design() -> dict:
    return json.loads(DESIGN_PATH.read_text(encoding="utf-8"))


def method_color(key: str) -> str:
    colors = design()["methods"]
    if key not in colors:
        raise KeyError(f"unknown method key {key!r}; known keys: {sorted(colors)}")
    return colors[key]


def reference_color() -> str:
    """Gray for the declared reference (commanded motion, ground truth or a privileged comparison)."""
    return design()["reference"]


def apply_style() -> None:
    d = design()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": FONT_FAMILY,
        "font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "medium", "axes.labelsize": 9.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False,
        "text.color": d["text"], "axes.labelcolor": d["text"],
        "axes.edgecolor": d["muted"], "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": d["grid"], "grid.linewidth": 0.5,
        "xtick.color": d["muted"], "ytick.color": d["muted"],
        "xtick.labelcolor": d["text"], "ytick.labelcolor": d["text"],
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "axes.prop_cycle": cycler(color=list(d["methods"].values())),
        "figure.facecolor": d["surface"], "axes.facecolor": d["surface"], "savefig.facecolor": d["surface"],
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.format": "svg",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "svg.fonttype": "none",  # keep text as text: page fonts apply and labels stay searchable
    })
