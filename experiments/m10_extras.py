"""Milestone 10 report extras: the success-criterion table (every family and group composite of each
learned method against the best classical controller), the latency table and the torque-rate column."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from report_api import table  # noqa: E402

from experiments.d7_extras import label, mean_metric  # noqa: E402
from experiments.d_report import CLASSICAL_KEYS, GROUP_LABELS, REGIME_SHORT, add_best_classical  # noqa: E402

REPORT = Path("research/reports/15-m10-across-the-board")
FIG = REPORT / "figures"


def criterion_table(groups: dict, learned: list[str], tolerance: float = 0.05, caption: str | None = None) -> str:
    """Per group and family: the best classical controller's cap composite and each learned method's ratio to it;
    within the tolerance or better counts as met (ADR 0010 success criterion)."""
    rows, verdict = [], {k: {"met": 0, "total": 0} for k in learned}
    for gname, res in groups.items():
        if gname.startswith("latency"):
            continue
        add_best_classical(res)
        if "best_classical" not in res:
            continue
        for r in sorted({m["regime"] for m in res["best_classical"]}):
            ref = mean_metric(res["best_classical"], r, "composite_cap")
            src = next(m["best_classical_source"] for m in res["best_classical"] if m["regime"] == r)
            row = [GROUP_LABELS.get(gname, gname), REGIME_SHORT.get(r, r), f"{label(src)} ({ref:.2f})"]
            for k in learned:
                if k not in res:
                    row.append("—"); continue
                v = mean_metric(res[k], r, "composite_cap")
                ratio = v / ref if ref > 0 else float("nan")
                ok = ratio <= 1.0 + tolerance
                verdict[k]["total"] += 1; verdict[k]["met"] += int(ok)
                row.append(f"{v:.2f} ({100 * (ratio - 1):+.0f}%){'' if ok else ' ✗'}")
            rows.append(row)
    rows.append(["All", "families and groups", "met / total"] + [f"{verdict[k]['met']} / {verdict[k]['total']}" for k in learned])
    return table(["Group", "Family", "Best classical (cap composite)"] + [label(k) for k in learned], rows,
                 caption=caption or f"Success criterion of ADR 0010: the cap composite of each learned method against the best classical controller of the family (the lower of the tuned PDs and their feedforward variants); within {100 * tolerance:.0f}% or better counts as met, a cross marks a miss.", css_class="wide")


def latency_table(groups: dict, methods: list[str], caption: str | None = None) -> str:
    """In-distribution composites with and without the 5 ms compute latency, per learned method."""
    lat = next((g for g in groups if g.startswith("latency")), None)
    if lat is None:
        return ""
    regimes = ("hold", "free", "policy", "step", "contact2", "blocked", "task")
    rows = []
    for k in methods:
        if k not in groups[lat]:
            continue
        a, b = groups["in_distribution"][k], groups[lat][k]
        rows.append([label(k)] + [f"{mean_metric(a, r, 'composite_cap'):.2f} / {mean_metric(b, r, 'composite_cap'):.2f}" for r in regimes] + [f"{int(sum(m['diverged'] for m in b))}"])
    return table(["Controller"] + [REGIME_SHORT.get(r, r) + " (0 / 5 ms)" for r in regimes] + ["Diverged at 5 ms"], rows,
                 caption=caption or "Realism check: cap composites of the in-distribution families with the learned controllers at zero and at 5 ms of compute latency (decisions made every 10 ms take effect 5 ms later); classical controllers are unchanged.", css_class="wide")


def torque_rate_table(groups: dict, methods: list[str], caption: str | None = None) -> str:
    regimes = ("hold", "free", "policy", "step", "contact2", "blocked", "task", "push", "pick")
    rows = []
    for k in methods:
        vals = []
        for r in regimes:
            src = groups["tasks"] if r in ("push", "pick") else groups["in_distribution"]
            vals.append(f"{mean_metric(src[k], r, 'torque_rate_rms'):.0f}" if k in src else "—")
        rows.append([label(k)] + vals)
    return table(["Controller"] + [REGIME_SHORT.get(r, r) for r in regimes], rows, caption=caption or "Torque-rate RMS (Nm/s) per family: the smoothness column of ADR 0010.", numeric_cols=tuple(range(1, len(regimes) + 1)), css_class="wide")
