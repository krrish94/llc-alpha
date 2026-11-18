"""Tables of report 23 (Milestone 12, the consolidation on the torque-interface Panda; ADR 0012, protocol v5), built from
runs/m12/metrics_test.json: the fourteen families against the global classical set and the per-family PD, the criterion with
either reference, the deployment groups (base stiffness, actuator lag, compute latency), the depth comparison with paired
intervals, the data-scale control, and the task families' success and forces."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from report_api import compare_stats, table  # noqa: E402

from experiments.d7_extras import LABELS, label, load, mean_metric  # noqa: E402
from experiments.d_report import CLASSICAL_KEYS, GROUP_LABELS, REGIME_SHORT, add_best_classical, build_compare  # noqa: E402
from experiments.m10_extras import criterion_table  # noqa: E402

RUNS = Path("runs/m12")
FAMILIES = ("hold", "free", "policy", "step", "contact2", "blocked", "task", "push", "pick", "wipe", "erase", "peg", "drawer", "door")
NEW_FAMILIES = ("wipe", "erase", "peg", "drawer", "door")
STUDENTS = {"distill_consol_l4": "Student, 4 layers, at scale", "distill_consol_l10": "Student, 10 layers, at scale", "distill_consol_l4_small": "Student, 4 layers, Milestone 10 budget"}
LABELS.update({"pd_family": "Per-family PD + feedforward (tuned per family)", "pd_shared": "PD tuned on the tracking composite", "pd_shared_cap": "PD tuned on the cap composite",
               "pd_feedforward": "PD (tracking-tuned) + feedforward", "pd_shared_cap_feedforward": "PD (cap-tuned) + feedforward", "best_classical": "Best global classical controller of the family",
               "oracle_pd": "Oracle around the PD (privileged ceiling)", **STUDENTS})
GROUP_LABELS.update({"base_x0.5": "Base stiffness ×0.5 (learned controllers)", "base_x2": "Base stiffness ×2 (learned controllers)", "lag_3ms": "Actuator lag 3 ms and delay 5 ms",
                     "latency_5ms": "Compute latency 5 ms (learned controllers)", "latency_10ms": "Compute latency 10 ms (learned controllers)"})


def load_metrics(path: Path = RUNS / "metrics_test.json", reference_keys=CLASSICAL_KEYS):
    """{group: {method: rows}} with ``best_classical`` from ``reference_keys`` in every group."""
    data, groups = load(path)
    for g in groups.values():
        add_best_classical(g, keys=reference_keys)
    return data, groups


def with_reference(groups: dict, keys) -> dict:
    """A copy of the groups whose ``best_classical`` is chosen among ``keys`` (the per-family PD as the goal reference)."""
    out = {}
    for g, res in groups.items():
        res2 = {k: v for k, v in res.items() if k != "best_classical"}
        add_best_classical(res2, keys=keys)
        out[g] = res2
    return out


def _regimes(res: dict, key: str):
    return [r for r in FAMILIES if any(m["regime"] == r for m in res.get(key, []))]


def families_table(groups: dict, methods, caption: str | None = None) -> str:
    """Rows: the fourteen families (in distribution, then the task scenes); columns: the methods' cap composites."""
    rows = []
    for g in ("in_distribution", "tasks"):
        res = groups.get(g, {})
        first = next((k for k in methods if k in res), None)
        if first is None:
            continue
        for r in _regimes(res, first):
            rows.append([REGIME_SHORT.get(r, r)] + [f"{mean_metric(res[k], r, 'composite_cap'):.2f}" if k in res else "—" for k in methods])
    return table(["Family"] + [label(k) for k in methods], rows, caption=caption or "Cap composite per family (32 cases each; lower is better): the global classical set, the per-family PD, the oracle and the students.", css_class="wide")


def criterion_counts(groups: dict, learned, tolerance: float = 0.05) -> dict:
    out = {k: [0, 0] for k in learned}
    for gname, res in groups.items():
        if gname.startswith("latency") or "best_classical" not in res:
            continue
        for r in sorted({m["regime"] for m in res["best_classical"]}):
            ref = mean_metric(res["best_classical"], r, "composite_cap")
            for k in learned:
                if k in res:
                    v = mean_metric(res[k], r, "composite_cap")
                    out[k][1] += 1; out[k][0] += int(v <= (1 + tolerance) * ref)
    return out


def deployment_table(groups: dict, learned, caption: str | None = None) -> str:
    """The deployment groups against the in-distribution rows of the same families: the ratio of cap composites per family and the
    number of families within 10% (ADR 0012, criterion v)."""
    rows = []
    for g in ("base_x0.5", "base_x2", "lag_3ms", "latency_5ms", "latency_10ms"):
        if g not in groups:
            continue
        for k in learned + ["best_classical"]:
            if k not in groups[g] or k not in groups["in_distribution"]:
                continue
            cells, n_ok, n = [], 0, 0
            for r in _regimes(groups[g], k):
                a, b = mean_metric(groups["in_distribution"][k], r, "composite_cap"), mean_metric(groups[g][k], r, "composite_cap")
                ratio = b / a if a > 0 else float("nan")
                cells.append(f"{REGIME_SHORT.get(r, r).split(' (')[0]} {ratio:.2f}")
                n += 1; n_ok += int(ratio <= 1.10)
            div = int(sum(m.get("diverged", 0) > 0 for m in groups[g][k]))
            rows.append([GROUP_LABELS.get(g, g), label(k), "; ".join(cells), f"{n_ok} / {n}", str(div)])
    return table(["Group", "Controller", "Cap composite relative to the in-distribution row, per family", "Families within 10%", "Diverged"], rows,
                 caption=caption or "The deployment groups (ADR 0012): the same families with the learned controllers' base at half and double stiffness (the damping by the square root), with a 3 ms actuator lag and a 5 ms transport delay for every controller, and with a compute latency of 5 and 10 ms; each cell is the group's cap composite over the in-distribution one.", css_class="wide")


def depth_table(data: dict, groups: dict, caption: str | None = None) -> str:
    """The 10-layer student against the 4-layer one, per family: means and the paired 95% bootstrap interval of the difference."""
    a, b = "distill_consol_l4", "distill_consol_l10"
    rows = []
    for g in ("in_distribution", "tasks"):
        res = groups.get(g, {})
        if a not in res or b not in res:
            continue
        cmp = build_compare(res, [(a, a), (b, b)], a)
        for r in _regimes(res, a):
            st = {s["key"]: s for s in compare_stats(cmp, r, "composite_cap")}
            sa, sb = st[a], st[b]
            lo, hi = sb["diff_ci"]
            sig = "" if np.isnan(lo) or lo <= 0 <= hi else " *"
            rows.append([REGIME_SHORT.get(r, r), f"{sa['mean']:.2f}", f"{sb['mean']:.2f}", f"{sb['mean'] - sa['mean']:+.2f} [{lo:+.2f}, {hi:+.2f}]{sig}", f"{100 * (sb['mean'] / sa['mean'] - 1):+.0f}%" if sa["mean"] > 0 else "—"])
    return table(["Family", "4 layers", "10 layers", "Difference (10 minus 4), paired 95% interval", "Relative"], rows,
                 caption=caption or "Depth (ADR 0012, criterion iii): the 10-layer student against the 4-layer one on the cap composite, 32 cases per family, one seed each; the interval is the paired bootstrap over cases (* excludes zero).", numeric_cols=(1, 2, 3, 4))


def scale_table(groups: dict, caption: str | None = None) -> str:
    """The 4-layer student at scale against the same recipe at the Milestone 10 budget, per family and group (criterion iv)."""
    a, b = "distill_consol_l4_small", "distill_consol_l4"
    rows = []
    for g, res in groups.items():
        if g.startswith("latency") or a not in res or b not in res:
            continue
        for r in _regimes(res, a):
            va, vb = mean_metric(res[a], r, "composite_cap"), mean_metric(res[b], r, "composite_cap")
            ref = mean_metric(res["best_classical"], r, "composite_cap") if "best_classical" in res else float("nan")
            rows.append([GROUP_LABELS.get(g, g).split(" (")[0], REGIME_SHORT.get(r, r), f"{ref:.2f}", f"{va:.2f}", f"{vb:.2f}", f"{100 * (vb / va - 1):+.0f}%" if va > 0 else "—"])
    return table(["Group", "Family", "Best global classical", "4 layers, 576 episodes", "4 layers, 6400 episodes", "Change"], rows,
                 caption=caption or "The data scale (ADR 0012, criterion iv): the same recipe, warm start and distribution at the Milestone 10 budget (12 iterations of 48 episodes) and at scale (20 of 320); cap composites per family and group.", css_class="wide", numeric_cols=(2, 3, 4, 5))


def task_table(groups: dict, methods, caption: str | None = None) -> str:
    """The task families: success rate, the force above the cap, the sustained peak, per method."""
    res = groups.get("tasks", {})
    rows = []
    for r in [r for r in ("push", "pick", "peg", "drawer", "door", "wipe", "erase") if any(m["regime"] == r for m in res.get(methods[0], []))]:
        for k in methods:
            if k not in res:
                continue
            succ = mean_metric(res[k], r, "task_success") if r in ("push", "pick", "peg", "drawer", "door") else float("nan")
            if r in ("wipe", "erase"):
                extra = f"{mean_metric(res[k], r, 'press_deficit'):.1f} N deficit, {mean_metric(res[k], r, 'contact_loss_rate'):.1f} losses/s"
            elif r == "peg":
                extra = f"{mean_metric(res[k], r, 'insertion_depth') * 1e3:.0f} mm inserted"
            elif r in ("drawer", "door"):
                extra = f"travel {mean_metric(res[k], r, 'travel'):.3f}"
            elif r == "pick":
                extra = f"grasps {100 * mean_metric(res[k], r, 'grasp_success'):.0f}%, {1e3 * mean_metric(res[k], r, 'pregrasp_disturbance'):.1f} mm moved and {mean_metric(res[k], r, 'grasp_force_peak_20ms'):.1f} N on the block before the grasp, placed within {1e3 * mean_metric(res[k], r, 'place_error'):.0f} mm"
            else:
                extra = f"{1e3 * mean_metric(res[k], r, 'push_error'):.0f} mm short, {np.degrees(mean_metric(res[k], r, 'push_yaw')):.0f} deg yaw"
            rows.append([REGIME_SHORT.get(r, r), label(k), f"{100 * succ:.0f}%" if np.isfinite(succ) else "—", f"{mean_metric(res[k], r, 'excess_force_cap'):.1f}", f"{mean_metric(res[k], r, 'f_peak_20ms'):.0f}", f"{mean_metric(res[k], r, 'torque_rate_rms'):.0f}", extra, f"{mean_metric(res[k], r, 'composite_cap'):.2f}"])
    return table(["Family", "Controller", "Success", "Force above the cap (N)", "Sustained peak (N)", "Torque rate (Nm/s)", "Family metric", "Composite"], rows,
                 caption=caption or "The task families of protocol v5 (32 cases each): success (the peg inserted 20 mm, the drawer within 5 mm of its stop, the door within 2 degrees, the pushed box and the placed block as in v4), the mean force above the 15 N cap while in contact, the sustained peak, the torque rate; wiping and erasing report the press deficit and the contact losses.", css_class="wide")
