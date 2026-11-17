"""Handoff page (research/reports/18-handoff): a standalone executive summary of the project so far.

Written for a reader who takes the work over: the question and the setting, the controller that
resulted and how it is trained, the whole hypothesis space with what worked and what did not, the
gaps, and what to revisit when the method is scaled to more data, iterations and compute. The
headline numbers and the criterion grid are computed from the round-2 benchmark
(runs/m10r2/metrics_test.json); everything else is a record of reports 01 to 17.

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/handoff.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, design  # noqa: E402
from report_api import esc, figure, table  # noqa: E402

from experiments.d7_extras import load, mean_metric  # noqa: E402
from experiments.d_report import add_best_classical  # noqa: E402

REPORT = Path("research/reports/18-handoff")
FIG = REPORT / "figures"
BEST = "best_classical"
STUDENT = "distill_v3_r2_two_teachers"
STUDENT_ORACLE = "distill_v3_r2_gain_ext"
STUDENT_M10 = "distill_v3_gain_ext"
ORACLE = "oracle_pd"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

GROUP_ROWS = [("in_distribution", "In distribution"), ("heldout_workspace", "Held-out workspace"), ("speed_x1.5", "Speed ×1.5"), ("speed_x2", "Speed ×2"),
              ("payload_3kg", "Payload 3 kg"), ("ood_instances", "Held-out instances"), ("surface_2000", "Surface 2 kN/m"), ("surface_500", "Surface 500 N/m"), ("tasks", "Tasks")]
FAMILY_COLS = [("hold", "Hold"), ("free", "Free motion"), ("policy", "Policy-like"), ("step", "Step"), ("contact2", "Press, slide"), ("blocked", "Blocked"), ("task", "Task"), ("push", "Push"), ("pick", "Pick")]
SOURCE_LABEL = {"pd_shared": "PD, tracking-tuned", "pd_shared_cap": "PD, cap-tuned", "pd_feedforward": "PD, tracking-tuned, with feedforward", "pd_shared_cap_feedforward": "PD, cap-tuned, with feedforward"}


def rep(rid: str, text: str) -> str:
    return f'<a href="../{rid}/index.html">{text}</a>'


def page(pid: str, text: str) -> str:
    return f'<a href="../{pid}/index.html">{text}</a>'


def raw_table(headers, rows, caption: str | None = None, css_class: str | None = None) -> str:
    """report_api.table without cell escaping: cells carry links and emphasis (headers are escaped)."""
    lines = [f'<table class="{esc(css_class)}">' if css_class else "<table>"]
    if caption:
        lines.append(f"<caption>{caption}</caption>")
    lines.append("<thead>\n<tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr>\n</thead>\n<tbody>")
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row has {len(row)} cells, expected {len(headers)}")
        lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    lines.append("</tbody>\n</table>")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ figures
def _tint(color: str, t: float) -> tuple[float, float, float]:
    c = np.array(to_rgb(color))
    return tuple(1.0 - t * (1.0 - c))


def fig_criterion_grid(groups: dict) -> str:
    """Ratio of each learned controller's cap composite to the best classical controller's, per family and group,
    for the three students of Milestone 10 (rounds 1 and 2): the criterion of ADR 0010 as a grid."""
    apply_style()
    d = design()
    panels = [(STUDENT_M10, "Milestone 10 student\n(gravity base, oracle teacher)"), (STUDENT_ORACLE, "Round-2 student\n(feedforward base, oracle teacher)"), (STUDENT, "Two-teacher student\n(the handoff controller)")]
    cmap = LinearSegmentedColormap.from_list("criterion", [(0.0, _tint(d["ok"], 0.55)), (0.5, to_rgb(d["grid"])), (1.0, _tint(d["warn"], 0.55))])
    lim = np.log2(2.5)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        grid = np.full((len(GROUP_ROWS), len(FAMILY_COLS)), np.nan)
        met = total = 0
        for i, (g, _) in enumerate(GROUP_ROWS):
            res = groups[g]
            for j, (fam, _) in enumerate(FAMILY_COLS):
                if not any(m["regime"] == fam for m in res[BEST]) or key not in res:
                    continue
                ref = mean_metric(res[BEST], fam, "composite_cap")
                grid[i, j] = mean_metric(res[key], fam, "composite_cap") / ref
                total += 1
                met += int(grid[i, j] <= 1.05)
        shown = np.clip(np.log2(grid), -lim, lim)
        masked = np.ma.masked_invalid(shown)
        ax.imshow(masked, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}" + ("" if grid[i, j] <= 1.05 else " ✗"), ha="center", va="center", fontsize=8, color=d["text"])
        ax.set_xticks(range(len(FAMILY_COLS))); ax.set_xticklabels([c[1] for c in FAMILY_COLS], rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(GROUP_ROWS))); ax.set_yticklabels([r[1] for r in GROUP_ROWS], fontsize=8.5)
        ax.set_title(f"{title}\n{met} of {total} rows met", fontsize=9.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim, lim))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, ticks=np.log2([0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 2.5]))
    cbar.ax.set_yticklabels(["0.4", "0.5", "0.7", "1.0", "1.4", "2.0", "2.5"], fontsize=8)
    cbar.set_label("cap composite relative to the best classical controller (log scale; lower is better)", fontsize=8.5)
    cbar.outline.set_visible(False)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "criterion_grid.svg", bbox_inches="tight"); plt.close(fig)
    return figure("figures/criterion_grid.svg", "The pre-registered criterion of ADR 0010 as a grid: the cap composite of each learned controller divided by that of the best classical controller of the family, per held-out group (16 test cases per family, three seeds pooled; protocol v3 on the Panda with the Robotiq 2F-85). A ratio at or below 1.05 counts as met; a cross marks a miss. Blank cells are families a group does not contain. Left to right: the Milestone 10 student (report 15), the round-2 student distilled from the oracle alone, and the two-teacher student (report 17).", "fig-criterion-grid", wide=True)


def architecture_figure() -> str:
    """Inline SVG: the controller at run time and the training pipeline."""
    d = design()
    W, H = 1000, 470
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" role="img" aria-labelledby="fig-arch-title" style="font-family:{FONT};font-size:12px;max-width:100%;height:auto">',
           '<title id="fig-arch-title">The learned low-level controller at run time and its training pipeline</title>',
           f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{d["muted"]}"/></marker></defs>']

    def box(x, y, w, h, lines, stroke=d["separator"], fill=d["surface_quiet"], bold_first=True, size=12):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        lh = size + 3
        y0 = y + h / 2 - lh * (len(lines) - 1) / 2 + 4
        for k, line in enumerate(lines):
            weight = ' font-weight="600"' if (k == 0 and bold_first) else ""
            out.append(f'<text x="{x + w / 2}" y="{y0 + k * lh:.1f}" text-anchor="middle" fill="{d["text"]}" font-size="{size}"{weight}>{esc(line)}</text>')

    def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None):
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{d["muted"]}" stroke-width="1.2" marker-end="url(#arrow)"/>')
        if label:
            out.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" fill="{d["muted"]}" font-size="11">{esc(label)}</text>')

    def heading(x, y, text):
        out.append(f'<text x="{x}" y="{y}" fill="{d["text"]}" font-size="13" font-weight="600">{esc(text)}</text>')

    student = d["methods"].get(STUDENT, d["selection"])
    # --- run time
    heading(20, 24, "A. The controller at run time (every signal a Panda exposes)")
    box(20, 52, 180, 64, ["Reference", "q_ref, q̇_ref from the policy", "or trajectory, any rate"])
    box(250, 44, 330, 80, ["Base law at 1 kHz", "τ = α ∘ (K_p e + K_d ė)", "+ M(q) q̈_ref + C(q, q̇_ref) q̇_ref + g(q)", "+ Δτ        (nominal model; e = q_ref − q)"])
    box(640, 52, 180, 64, ["Robot", "Panda + Robotiq 2F-85", "torque interface, 1 kHz"])
    arrow(200, 84, 248, 84)
    arrow(580, 84, 638, 84, "τ", 609, 78)
    out.append(f'<path d="M 730 116 L 730 140 L 415 140" fill="none" stroke="{d["muted"]}" stroke-width="1.2"/>')
    arrow(415, 140, 415, 126, "q, q̇ (feedback)", 560, 154)
    box(560, 176, 300, 72, ["History, two scales", "100 ms at 10 ms spacing + last 32 ms at 1 ms", "of q, q̇, commanded τ and the external-torque", "estimate τ_ext = τ_link − model"], bold_first=True)
    out.append(f'<path d="M 790 116 L 790 160 L 710 160" fill="none" stroke="{d["muted"]}" stroke-width="1.2"/>')
    arrow(710, 160, 710, 174)
    box(120, 176, 360, 72, ["Network, decision every 10 ms", "2 × 128 MLP; compute latency 5 ms modelled, output held", "α: per-joint gain multipliers, K_p × [0.02, 1.5], K_d × [0.3, 2]", "Δτ: residual, bounded at half the torque limit"], stroke=student)
    arrow(560, 212, 482, 212)
    arrow(300, 176, 300, 126, "α, Δτ", 322, 156)
    out.append(f'<path d="M 110 84 L 110 212 L 118 212" fill="none" stroke="{d["muted"]}" stroke-width="1.2" stroke-dasharray="3 3"/>')
    out.append(f'<text x="60" y="150" fill="{d["muted"]}" font-size="11">current reference</text>')
    # --- training
    heading(20, 296, "B. Training, entirely in simulation (MuJoCo; robot instances, payloads of 0 to 3 kg, sensor noise per window)")
    box(20, 322, 170, 112, ["0. Shared PD gains", "CMA-ES on the tracking", "composite, training split", "(40 iterations, 4 cases", "per family)", "report 14"], size=11)
    box(210, 322, 270, 112, ["1. Residual: actor-critic through the plant", "finite-difference MuJoCo gradients, clipped", "at the batch median; bounded tracking kernels", "(20 mrad), light torque-rate penalty, critic;", "free-space families only, 250 iterations,", "selected on free space; reports 09, 17"], stroke=d["methods"].get("pd_residual_v3_ff", d["separator"]), size=11)
    box(500, 322, 300, 112, ["2. Student: DAgger from two privileged teachers", "free space: ideal-robot torque, relative to the", "nominal law; contact: the impedance oracle", "(7 N + 500 N/m × e, capped at 15 N; stop and wait", "when blocked); free-space decisions ×3;", "12 iterations × 48 episodes; reports 13, 17"], stroke=student, size=11)
    box(820, 322, 160, 112, ["3. Selection, benchmark", "best checkpoint on the", "validation cap composite;", "protocol v3: 33 rows,", "three seeds, 5 ms latency", "reports 15, 16, 17"], size=11)
    arrow(190, 378, 208, 378)
    arrow(480, 378, 498, 378)
    arrow(800, 378, 818, 378)
    out.append(f'<text x="20" y="458" fill="{d["muted"]}" font-size="11">Gray boxes: classical or fixed parts. Coloured outlines: learned parts (green: the two-teacher student; blue: its stage-1 base). Stage 1 inherits the gains of stage 0; stage 2 is warm-started from stage 1 with the gain head at neutral.</text>')
    out.append("</svg>")
    caption = ("The controller that resulted (A) and how it is trained (B). At run time a 1 kHz PD law with the nominal inverse-dynamics feedforward runs under a network that every 10 ms sets bounded per-joint gain multipliers and a bounded torque residual from two-scale proprioceptive history and the external-torque estimate. "
               "Training uses only simulation: shared gains by CMA-ES, a torque residual by backpropagation through the plant on free-space references, then imitation of two privileged teachers, with checkpoint selection on a validation split and the benchmark on a disjoint test split.")
    return f'<figure class="wide" id="fig-architecture">\n{chr(10).join(out)}\n<figcaption>{caption}</figcaption>\n</figure>'


# ------------------------------------------------------------------------------------------ tables
SINGLE = "pd_feedforward"  # the best single classical controller across the 33 rows (computed by single_controller_stats; stated in the table caption)
CLASSICAL = ("pd_shared", "pd_shared_cap", "pd_feedforward", "pd_shared_cap_feedforward")
SOURCE_SHORT = {"pd_shared": "tracking-tuned", "pd_shared_cap": "cap-tuned", "pd_feedforward": "tracking-tuned + feedforward", "pd_shared_cap_feedforward": "cap-tuned + feedforward"}
HEADLINE_ROWS = [  # (label, group, regime, metric, scale, decimals, oracle shown)
    ("Free motion, position RMSE (mrad)", "in_distribution", "free", "pos_rmse", 1e3, 1, False),
    ("Free motion with a 3 kg payload, position RMSE (mrad)", "payload_3kg", "free", "pos_rmse", 1e3, 1, False),
    ("Hold with 10 N pulses, position RMSE (mrad)", "in_distribution", "hold", "pos_rmse", 1e3, 1, False),
    ("Policy-like targets, position RMSE (mrad)", "in_distribution", "policy", "pos_rmse", 1e3, 1, False),
    ("Policy-like targets, torque-rate RMS (Nm/s)", "in_distribution", "policy", "torque_rate_rms", 1.0, 0, False),
    ("Step targets, settling time (s)", "in_distribution", "step", "settle_time", 1.0, 2, False),
    ("Press and slide, mean force (N; band 7 to 12)", "in_distribution", "contact2", "f_mean", 1.0, 1, True),
    ("Press and slide, force above the 15 N cap (N)", "in_distribution", "contact2", "excess_force_cap", 1.0, 1, True),
    ("Press and slide, force below the 7 N floor (N)", "in_distribution", "contact2", "press_deficit", 1.0, 2, True),
    ("Blocked target, force above the cap (N)", "in_distribution", "blocked", "excess_force_cap", 1.0, 1, True),
    ("Blocked target, apparent stiffness (N/m)", "in_distribution", "blocked", "apparent_stiffness", 1.0, 0, True),
    ("Blocked target, drift from the contact point (mm)", "in_distribution", "blocked", "blocked_drift", 1e3, 1, True),
    ("Blocked target, release reaction (s)", "in_distribution", "blocked", "release_reaction", 1.0, 2, True),
    ("Blocked target on a 500 N/m surface, contact stiffness (N/m)", "surface_500", "blocked", "contact_impedance", 1.0, 0, True),
    ("Task sequence, force above the cap (N)", "in_distribution", "task", "excess_force_cap", 1.0, 1, True),
    ("Box pushing, success (%)", "tasks", "push", "task_success", 100.0, 0, False),
    ("Pick-and-place, success (%)", "tasks", "pick", "task_success", 100.0, 0, False),
    ("Pick-and-place, force on the table while placing (N)", "tasks", "pick", "place_force_peak_20ms", 1.0, 1, False),
    ("Pick-and-place, lift tracking after the release (mrad)", "tasks", "pick", "lift_rmse", 1e3, 1, False),
    ("Free motion, torque-rate RMS (Nm/s)", "in_distribution", "free", "torque_rate_rms", 1.0, 0, False),
]


def criterion_rows(groups: dict) -> list[tuple[str, str, dict]]:
    """(group, family, {method: cap composite}) for the 33 rows of the criterion."""
    out = []
    for g, res in groups.items():
        if g.startswith("latency"):
            continue
        for r in sorted({m["regime"] for m in res[BEST]}):
            out.append((g, r, {k: mean_metric(res[k], r, "composite_cap") for k in (*CLASSICAL, STUDENT, BEST) if k in res}))
    return out


def single_controller_stats(groups: dict) -> dict:
    """Per single classical controller: rows within 5% of the per-family best, mean ratio to it, and the student's rows met against it."""
    rows = criterion_rows(groups)
    stats = {}
    for k in CLASSICAL:
        stats[k] = {"within": sum(v[k] <= 1.05 * v[BEST] for _, _, v in rows), "ratio": sum(v[k] / v[BEST] for _, _, v in rows) / len(rows),
                    "student_met": sum(v[STUDENT] <= 1.05 * v[k] for _, _, v in rows), "total": len(rows)}
    return stats


def headline_numbers(groups: dict) -> dict:
    """The headline table's rows (per method) and the scalars the summary quotes."""
    lat = next(g for g in groups if g.startswith("latency"))
    fam = ("hold", "free", "policy", "step", "contact2", "blocked", "task")
    idd = groups["in_distribution"]
    lat_change = max(abs(mean_metric(groups[lat][STUDENT], r, "composite_cap") / mean_metric(idd[STUDENT], r, "composite_cap") - 1.0) for r in fam)
    rows = []
    for label_text, g, r, key, scale, dec, oracle in HEADLINE_ROWS:
        res = groups[g]
        src = next(m["best_classical_source"] for m in res[BEST] if m["regime"] == r)
        rows.append({"label": label_text, "dec": dec, "src": SOURCE_SHORT.get(src, src), "best": scale * mean_metric(res[BEST], r, key), "single": scale * mean_metric(res[SINGLE], r, key),
                     "student": scale * mean_metric(res[STUDENT], r, key), "oracle": scale * mean_metric(res[ORACLE], r, key) if oracle else None})
    by = {row["label"]: row for row in rows}
    n = {"lat_change": 100 * lat_change, "rows": rows, "single": single_controller_stats(groups)}
    for name, label_text in (("free", "Free motion, position RMSE (mrad)"), ("hold", "Hold with 10 N pulses, position RMSE (mrad)"), ("blocked_exc", "Blocked target, force above the cap (N)"),
                             ("place_f", "Pick-and-place, force on the table while placing (N)")):
        n[f"{name}_best"], n[f"{name}_student"] = by[label_text]["best"], by[label_text]["student"]
    return n


def headline_table(n: dict) -> str:
    def f(v, dec):
        return "—" if v is None else f"{v:.{dec}f}"

    rows = [[row["label"], f"{f(row['best'], row['dec'])} ({row['src']})", f(row["single"], row["dec"]), f(row["student"], row["dec"]), f(row["oracle"], row["dec"])] for row in n["rows"]]
    rows.append(["Largest change of a family composite at 5 ms compute latency (%)", "0 (1 kHz law)", "0 (1 kHz law)", f"{n['lat_change']:.1f}", "—"])
    st = n["single"]
    others = ", ".join(f"{SOURCE_SHORT[k]} {st[k]['within']}" for k in CLASSICAL if k != SINGLE)
    caption = (f"Headline numbers of the handoff controller against the classical set and the privileged oracle, protocol v3 on the Panda with the Robotiq 2F-85, 16 test cases per family, three seeds pooled (runs/m10r2/metrics_test.json). "
               f"'Best classical of the family' is a selection among the four fixed classical controllers, each one gain set for all families, named in parentheses. 'Best single classical controller' is the one of the four that comes closest to that selection across the 33 criterion rows, the tracking-tuned PD with feedforward "
               f"(within 5% of the per-family best on {st[SINGLE]['within']} of {st[SINGLE]['total']} rows; the others: {others}). The oracle changes only the contact response; its free-space rows are those of the PD. Paired bootstrap intervals are in reports 15 and 17.")
    return table(["Quantity (in distribution unless stated)", "Best classical of the family", "Best single classical controller", "Two-teacher student", "Oracle (privileged ceiling)"], rows, numeric_cols=(1, 2, 3, 4), caption=caption, css_class="wide")


def verdict_single(n: dict) -> str:
    st = n["single"][SINGLE]
    return (f"<p>Against one classical controller for everything, the tracking-tuned PD with feedforward, the student meets {st['student_met']} of {st['total']} rows with the same five policy-like misses, because on that family the feedforward PD is the per-family best anyway; "
            f"against the cap-tuned PD alone it meets all {st['total']}. The criterion of ADR 0010 is the harder one, against the per-family selection, and the grid above reports that.</p>")


def protocol_table() -> str:
    rows = [["Hold", "hold a configuration; a 10 N tool pulse of 0.2 s", "small error and deflection, smooth torque"],
            ["Free motion", "minimum-jerk joint paths between random configurations", "accurate, smooth"],
            ["Policy-like targets", "held targets at 10 to 50 Hz with jitter and jumps", "track the intent, no chatter"],
            ["Step targets", "joint-space steps", "fast settling, no overshoot"],
            ["Press and slide", "approach, press 0.4 to 1.4 cm (7 to 12 N), slide, release; table or wall", "force in the band, contact kept, no impact"],
            ["Blocked target", "a path 3 to 8 cm through the table or the wall", "stop at the surface and wait, resume when the target returns"],
            ["Task sequence (held out)", "reach, press, slide, lift, transport, press, lift", "the press band under a task-like sequence"],
            ["Box pushing", "push a 5 cm box 10 cm with the fingertip faces", "box within 2 cm and 15 degrees, gently"],
            ["Pick-and-place", "carry a held block, place it, release, lift", "place within 2 cm, upright, gently; track after the payload change"]]
    return table(["Family", "Reference", "Wanted"], rows, caption="Protocol v3 (ADR 0009): 16 test cases per family, each with a robot instance, a payload of 0 to 2 kg and sensor noise; a disjoint training split generates the training references and a validation split selects checkpoints. Held-out groups: workspace, speed ×1.5 and ×2, payload 3 kg, doubled instance deviations, compliant surfaces of 2 kN/m and 500 N/m; a 5 ms compute-latency group for the learned controllers. Each family has a cap composite, a weighted sum of its metrics normalised by reference scales with the force terms at the 15 N cap, lower is better.")


def matrix_table() -> str:
    r = rep
    rows = [["Learned PD gains (gain schedule)",
             "halves the torque rate on policy-like targets, soft everywhere (" + r("02-known-model", "02") + "); the most robust to delay and combined changes (" + r("03-robustness", "03") + ")",
             "matches training through the known model (" + r("04-learned-model", "04") + ")",
             "parked after protocol v1 (ADR 0005); returns as bounded multipliers inside the student (row 3)",
             "worse than backpropagation (" + r("05-ppo", "05") + ")", "—", "—"],
            ["PD + torque residual",
             "smooth tracking −36%, nothing else; the contact stage exploits the penalty contact model (" + r("02-known-model", "02") + ")",
             "parity with the known model at a third of the cost (" + r("04-learned-model", "04") + "); learned-model gradients stable through contact, lowest blocked excess of D3 (" + r("09-dd-gradients", "09") + ")",
             "<strong>kept.</strong> beats the tuned PD on every family and group except blocked targets (" + r("09-dd-gradients", "09") + "); the base of every student; on v3 with the feedforward base and free-space selection (" + r("17-m10-round-2", "17") + ")",
             "parity with its warm start; cuts forces by flinching at the 30 N limit (" + r("10-dd-ppo", "10") + "); small gains under the cap, apparent stiffness 1300 N/m (" + r("11-dd-contact", "11") + ")",
             "stop-and-wait learned from the oracle (" + r("13-dd-impedance", "13") + ")",
             "alone: wins hold and pushing, doubles the blocked composite (" + r("15-m10-across-the-board", "15") + ")"],
            ["Gain multipliers + residual", "—", "—",
             "gain head stays neutral when trained by the actor-critic under the cap reward (" + r("11-dd-contact", "11") + ")",
             "gain head never explored at exploration std 0.05 (" + r("11-dd-contact", "11") + ")",
             "<strong>kept.</strong> the student uses the gain head (0.82 in contact, 0.97 free) and lifts payloads (" + r("13-dd-impedance", "13") + ", " + r("15-m10-across-the-board", "15") + ")",
             "<strong>kept.</strong> as the free-space teacher of the two-teacher student: free motion 2.6 mrad (" + r("17-m10-round-2", "17") + ")"],
            ["Direct torque network",
             "worst of the three (" + r("02-known-model", "02") + "); least sensitive to observation noise (" + r("03-robustness", "03") + ")",
             "softest impacts, 37 N blocked excess, 56 mrad free motion (" + r("09-dd-gradients", "09") + ")",
             "collapses at the contact stage under finite-difference gradients (" + r("09-dd-gradients", "09") + "); as a free-space base wins hold, presses with 85 N (" + r("17-m10-round-2", "17") + ")",
             "untrainable (" + r("05-ppo", "05") + ")",
             "its student loses every free-space family: a torque held for 10 ms is open-loop (" + r("17-m10-round-2", "17") + ")",
             "not run"]]
    return raw_table(["Parameterisation", "Known differentiable model", "Learned dynamics model", "Plant gradients, actor-critic", "PPO in the plant", "Imitation of the privileged oracle", "Ideal-robot supervision"], rows,
                 caption="The hypothesis space of the agenda (three parameterisations by three learning routes, extended by the two supervised routes that the deep dives added), with the outcome of every cell that was tried and the report that holds the evidence. Kept: part of the handoff controller.", css_class="wide")


def chronology_table() -> str:
    r = rep
    rows = [[r("01-baselines", "01"), "Milestone 1", "Is there headroom over shared PD gains?", "Per-regime tuning cuts position error by 55 to 65% and torque rate by 73% on policy-like targets; the tuned sets are mutually incompatible.", "complete, protocol v1"],
            [r("02-known-model", "02"), "Milestone 2", "Do the three parameterisations capture it through a differentiable model?", "No. The residual improves smooth tracking only; the contact stage of training exploits the penalty contact model.", "negative result, investigated"],
            [r("03-robustness", "03"), "Milestone 3", "Robustness to changed dynamics and randomisation", "A 5 ms delay breaks the stiff PD (composite ×6.5), learned gains are unaffected; randomisation helps neural torque only.", "complete, v1"],
            [r("04-learned-model", "04"), "Milestone 4", "Can a learned model replace the known one as the training medium?", "Yes: 100 ms predictions within 2 to 14 mrad, gradient cosine 0.96, equal closed-loop results at a third of the cost.", "complete, v1"],
            [r("05-ppo", "05"), "Milestone 5", "PPO against the model-based routes", "Worse for every parameterisation; neural torque untrainable; 4.5 M plant steps per run.", "negative, budget-limited"],
            [r("06-phases", "06"), "Milestone 6", "Phase-scheduled gains on multi-phase trajectories", "Scheduling cuts impact peaks (241 to 82 N) but spikes torque at switches and fails with a payload.", "complete, v1"],
            [r("07-dd-baselines", "07"), "Deep dive D1", "Fixed PD under protocol v2 (instances, payloads, blocked targets)", "No fixed gain set is both accurate and compliant; the soft set sags under payloads (hold 16.7 against 2.8 mrad).", "complete, v2; two errata"],
            [r("08-dd-controllers", "08"), "Deep dive D2", "1 kHz residual and neural torque through MuJoCo gradients", "They infer the payload from the history (hold 3.4 to 5 mrad) but are not compliant on blocked targets; force terms destabilise training.", "seed 0 only"],
            [r("09-dd-gradients", "09"), "Deep dive D3", "Actor-critic, median-clipped gradients, learned-model gradients", "The residual beats the tuned PD on every family and group; its blocked excess force equals the PD's. The recipe of the base.", "complete, v2"],
            [r("10-dd-ppo", "10"), "Deep dive D4", "PPO in the plant from the D3 residual", "Parity with the warm start; forces cut by a learned flinch; contact maintenance not improved; two-case validation misranks.", "closed"],
            [r("11-dd-contact", "11"), "Deep dive D5", "Contact maintenance under a 15 N cap", "Three harness defects fixed; PPO under the cap pushes less but stays stiff; the gain head is never explored.", "stopped by the advisor; blocked target redefined"],
            [r("12-checkpoint-1", "12"), "Checkpoint 1", "Standalone summary", "The learned residual beats the tuned PD everywhere and limits forces; it neither presses to a target nor waits when blocked.", "checkpoint"],
            [r("13-dd-impedance", "13"), "Deep dive D6", "Impedance by imitation of a privileged oracle", "The oracle's law distilled by DAgger gives stop-and-wait (57 to 15 N above the cap; 527 N/m); the external torque is the contact cue; PPO adds nothing.", "complete, v2"],
            [r("14-m9-plant-v3", "14"), "Milestone 9", "Plant v3 (gripper), protocol v3, task families", "Baselines on the gripper plant; the sphere-plant student transfers its contact behaviour; light-contact detachment persists on the pads.", "complete, v3"],
            [r("15-m10-across-the-board", "15"), "Milestone 10", "One learned controller against the best classical, across the board", "27 of 33 rows met; every miss is free motion against the feedforward PD (11.9 against 4.9 mrad).", "complete, v3"],
            [r("16-checkpoint-2", "16"), "Checkpoint 2", "Standalone summary (revised after round 2)", "The two-teacher student as the headline controller.", "checkpoint"],
            [r("17-m10-round-2", "17"), "Milestone 10, round 2", "The free-motion gap, a torque network, two teachers", "Feedforward base, free-space base and 7 N floor: 29 rows; two teachers: 2.6 mrad free motion, 28 rows; the torque network fails as a student.", "complete, v3"]]
    return raw_table(["Report", "Stage", "Question", "Result", "Status"], rows, caption="Chronology of the seventeen reports (2026-11-15 to 2026-11-16). Protocol v1 numbers (nominal robot, sphere tool) and v2 numbers (instances, sphere tool) are not comparable with v3 numbers (instances, gripper).", css_class="wide")


def cost_table() -> str:
    rows = [["Shared PD gains, CMA-ES", "40 iterations, 4 cases per family", "13 400 episodes", "20 min"],
            ["Stage 1: residual by actor-critic through the plant", "250 iterations × 24 windows of 2 s; finite-difference gradients", "12 M plant steps (plus the finite-difference evaluations)", "57 min with six runs sharing the machine (40 min for the torque network)"],
            ["Stage 2: student by DAgger", "12 iterations × 48 episodes; 4 epochs per iteration", "270 to 300 k labelled decisions (cap 300 k)", "11 to 13 min per seed"],
            ["Benchmark, protocol v3 in full", "448 episodes per controller (10 groups), 23 controllers", "10 300 episodes", "21 min with 12 workers"],
            ["PPO in the plant (D4, for comparison)", "300 updates × 64 windows of 2 s", "38 M plant steps per run", "18 min alone, 3.6 s per update; 57 s per update with sixteen runs"],
            ["Backpropagation through the torch model (M2, for comparison)", "360 iterations, batch 24, 2.5 ms model step", "model steps only", "tens of minutes"]]
    return table(["Stage", "Configuration", "Simulated data", "Wall-clock on the laptop (M5 Pro, 15 cores, CPU MuJoCo)"], rows,
                 caption="What a run costs on the development machine. Every learned result of the handoff controller was produced by the first four rows, three seeds each; the figures are from the run logs of runs/m9 and runs/m10r2.", css_class="wide")


# ------------------------------------------------------------------------------------------ text
def summary(n: dict) -> str:
    return f"""<p><strong>State of the investigation.</strong> The project asked whether the low-level feedback loop of a torque-controlled manipulator can be learned so that one controller is competitive with, or better than, the best hand-tuned classical law in every operating regime at once. After seventeen reports on a simulated Franka Panda the answer is yes on 28 of the 33 pre-registered family-and-group rows, with the five misses on one family at 5 to 6%. The controller that does this does not replace the classical loop; it adds to it. A 1 kHz PD law with the nominal inverse-dynamics feedforward runs under a network that, every 10 ms, sets bounded per-joint gain multipliers and a bounded torque residual from 100 ms of proprioceptive history and an estimate of the external torque. It is trained in two stages, both in simulation: a residual by backpropagation through the plant on free-space references, then imitation of two privileged teachers, the ideal-robot torque in free space and a contact-impedance law in contact. It tracks free motion at {n['free_student']:.1f} mrad against {n['free_best']:.1f} for the feedforward PD, holds at {n['hold_student']:.1f} against {n['hold_best']:.1f}, waits at a blocked surface with {n['blocked_exc_student']:.0f} N above the 15 N cap where the best classical pushes {n['blocked_exc_best']:.0f} N, presses inside the 7 to 12 N band, places a block at {n['place_f_student']:.0f} N against {n['place_f_best']:.0f}, and moves by at most {n['lat_change']:.0f}% under a 5 ms compute latency.</p>
<p>Of the agenda's hypothesis space, three parameterisations by three learning routes, one cell survived: the residual form, trained by gradients through the plant and finished by imitation of a specified behaviour. Reinforcement learning (five rounds), direct torque networks and gain schedules on their own did not earn a place; a learned dynamics model did, as a training medium, but was not carried to the final plant. The remaining gaps are a smoothness miss on policy-like targets, a press band that sits 5 N above the cap on sliding contact, task-success rates on sixteen cases, privileged labels without a hardware path yet, and every claim resting on three seeds and sixteen cases per family. Sections 6 to 8 list the gaps, what to revisit when the method is scaled, and what a run costs.</p>"""


SETTING = """<h3>1.1 The question</h3>
<p>Every torque-controlled robot closes a feedback loop between the policy's targets and the motors, usually a PD or impedance law at 100 Hz to 1 kHz with gains tuned by hand. The {agenda} argues that this loop should be learned: no gain set serves every task, every phase of a task and every robot instance; policies treat the loop as a black box and pay for it in jerky and unstable behaviour; retuning by hand does not scale across robots, tasks and contact modes, and switching gain sets stably is hard in dynamic tasks; and a controller learned from proprioception in simulation has a small sim-to-real gap. The agenda's design space is three parameterisations (learned PD gains, PD plus a residual, direct torque prediction) by three learning routes (a known differentiable model, a learned dynamics model, reinforcement learning). Section 3 maps what was tried in it and what was added to it.</p>
<h3>1.2 The plant</h3>
<p>MuJoCo at 1 kHz with torque actuators at the datasheet limits. Two plants: v2, the Panda with a 2 cm spherical tool (reports 07 to 13), and v3, the Panda with the Robotiq 2F-85 baked closed into one rigid tool body (reports 14 to 17; opened to 38 mm for pick-and-place), each with a table and a wall under the explicit soft contact model of ADR 0001 (about 60 kN/m) and two compliant surfaces held out for evaluation. Every evaluation and training window draws a robot instance (link masses ±5%, inertias ±10%, centres of mass ±5 mm, joint damping 0.5 to 2×, armature ±20%, dry friction 0 to 0.3 Nm, actuator gain ±3%), a payload of 0 to 2 kg (3 kg in training and in one held-out group) and sensor noise. Every controller, classical or learned, uses the datasheet model only; the simulated plant's parameters are never read.</p>
<h3>1.3 The protocol</h3>
{protocol}
<h3>1.4 Baselines, ceiling and criterion</h3>
<p><strong>Classical set.</strong> Four fixed controllers, each one gain set for all families: a PD plus gravity law whose gains CMA-ES tuned on the mean tracking composite over the seven families (four training cases each), a second set tuned on the cap composite with the same budget, and the inverse-dynamics feedforward variant of each (the computed-torque term of the reference added to the PD). No controller is tuned per family. 'The best classical controller of the family' is the one of the four with the lowest cap composite on that family and group; it is the reference of every comparison, and this bar is optimistic by construction: the selection is made per row on the test cases, whereas each learned controller is one network selected once on the validation split. Section 2.3 also reports the comparison against a single classical controller.</p>
<p><strong>Ceiling.</strong> The privileged contact-impedance oracle around the PD (ADR 0008): it knows the surface and the payload and executes the wanted contact law exactly. It marks what the contact families can give; it is never a baseline.</p>
<p><strong>Criterion</strong> (ADR 0010, pre-registered before Milestone 10): within 5% of the best classical or better on every family and held-out group composite (33 rows), better on the three contact families and both task-success rates, three seeds with paired intervals excluding zero where a win is claimed, and no divergence under a 5 ms compute latency.</p>"""

RECIPE = """<h3>2.1 The law</h3>
<p>At 1 kHz the commanded torque is τ = α ∘ (K_p e + K_d ė) + M(q) q̈_ref + C(q, q̇_ref) q̇_ref + g(q) + Δτ with e = q_ref − q, the shared gains K_p, K_d of the tracking-tuned PD, the nominal model's M, C and g, and q̈_ref the clipped central difference of q̇_ref (held targets contribute no feedforward). The per-joint multipliers α (stiffness in [0.02, 1.5], damping in [0.3, 2] of the shared gains) and the residual Δτ (bounded at half the torque limit) are the network's outputs, held for 10 ms between decisions; a 5 ms compute latency is modelled, so a decision observed at t acts from t + 5 ms. The network is a two-layer MLP of width 128 on the current reference and state and a two-scale history, 100 ms at 10 ms spacing plus the last 32 ms at 1 ms, of q, q̇, the commanded torque and the external-torque estimate (the measured link-side torque minus the nominal model's prediction). Every input is a signal the Panda exposes.</p>
<h3>2.2 Training</h3>
<p><strong>Stage 0, gains.</strong> Shared gains by CMA-ES on the tracking composite of four training cases per family (40 iterations).</p>
<p><strong>Stage 1, the base residual.</strong> A torque residual on the feedforward PD, trained by a short-horizon actor-critic through the plant: windows of 2 s with a random instance, payload and (in 30% of the hold and free windows) a 10 N pulse; a reward of bounded tracking kernels (20 mrad half-width) with a light torque-rate penalty (weight 0.003); a privileged critic (payload, contact force and a preview of the reference as inputs; no torque history); gradients by MuJoCo's finite-difference transition derivatives, clipped per sample at three times the batch median. The base trains on the free-space families only (hold, free motion, policy-like, step), 250 iterations of 24 windows, and its checkpoint is selected on the free-space cap composite of four validation cases per family (ADR 0010, amendment 2).</p>
<p><strong>Stage 2, the student.</strong> The base converted to the gain-multiplier form (head at neutral, exact warm start) with the external-torque input added, trained by DAgger: the teacher drives in the first iteration only, then the student drives and the teachers label every decision. In free space the label is the ideal-robot torque: the torque the sampled plant needs so that the nominal robot's response to the nominal law results, external loads included and referenced to the nominal law (the PD plus feedforward without the student's own residual); decisions where the nominal law saturates or the correction exceeds 30 Nm are dropped. In contact the label is the oracle's torque: the base's normal force replaced by a force-bounded spring on the actual error, 7 N + 500 N/m × e capped at 15 N, the first-contact point held when the reference is blocked, a 200 ms blend back on release, the payload's weight compensated. Free-space decisions enter the label loss with weight 3. Twelve iterations of 48 episodes (270 to 300 thousand labelled decisions); the checkpoint with the best validation cap composite (two cases per free-space family, eight per contact family) is kept. Three seeds of everything.</p>
<h3>2.3 Where it stands</h3>"""

VERDICT = """<p>The student meets 28 of 33 rows. It wins the three contact families by 13 to 60% (blocked targets at 0.40 to 0.51 of the best classical in every group), every free-space family except one (hold 0.47, free motion 0.36 to 0.90, steps 0.79), both compliant surfaces, the 3 kg payload group, and push success (52 against 31%, eight or nine of sixteen boxes against five); placement is at the ceiling for everyone (100%) with the student placing at a third of the force. The five misses are the policy-like family (held targets at 10 to 50 Hz) at 1.05 to 1.06 in every group: lower tracking error than the PD, higher torque rate. The same student distilled from the oracle alone meets 29 rows and misses free motion where the nominal model is accurate (6.9 against 4.9 mrad); the Milestone 10 student, without the model term in its base, met 27. No learned controller diverges anywhere, and the residual students move by at most 5% at 5 ms of compute latency. The free-motion win is a mean over cases, not a uniform one: per seed the student loses 4 to 6 of the 16 cases, all where the nominal model is accurate and the feedforward PD is already at 1 to 3 mrad, by up to a factor of 4.8 (5.3 against 1.1 mrad, a 47 mrad burst on one joint); it wins the rest, by the most where the model is wrong (18 to 2.4 mrad on the PD's worst case).</p>"""

OTHER_AXES = """<p>Beyond the parameterisation-by-route matrix, the deep dives explored four further axes.</p>
<ul>
<li><strong>Inputs.</strong> A proprioceptive history is the implicit system identification the agenda asked for and was kept throughout (100 ms at 10 ms spacing, later with 32 ms dense). The external-torque estimate from the robot model was added in D5 and is the contact cue (D6). A target preview (future targets in the observation) added nothing beyond the classical feedforward and was shelved; a target history halved the policy-like torque rate in D3 and was not carried; Fourier features and an output low-pass were planned as a representation study and not run; a tool-frame wrench was shelved for its sim-to-real gap.</li>
<li><strong>Rewards.</strong> Instantaneous force penalties (D2) destabilised training and were an optimisation failure as weighted; filtered force terms (D5) still cost free-space tracking unless detached from the gradient; the cap reward with contact-maintenance terms ranked a compliant exemplar first but no optimiser found that behaviour; the final recipe uses the reward only for free space and specifies contact as a law.</li>
<li><strong>Protocol and plant.</strong> Protocol v1 (nominal robot, six regimes), v2 (instances, payloads, blocked targets, the 15 N cap, stop-and-wait, release metrics), v3 (gripper, press band 7 to 12 N, link clearance, task families). Three harness defects were found by probes and corrected with errata: blocked references without a block (60% of them), link collisions on the wall branch, and an evaluation worker that ignored the case's perturbation.</li>
<li><strong>Classical additions.</strong> The inverse-dynamics feedforward of the reference halves the tracking-tuned PD's free-motion error (8.8 to 4.9 mrad) and was, until the two-teacher student, worth more than any learned part on that family; phase-scheduled gains (M6) cut impact peaks at the price of torque spikes and failed with a payload.</li>
</ul>"""

WORKED = """<ol>
<li><strong>A proprioceptive history identifies the payload.</strong> From the first 1 kHz residual on (D2), learned controllers held a loaded arm at 3.4 to 5 mrad where the soft PD sagged to 16.7; on the 3 kg held-out payload the handoff controller tracks free motion at 5.0 against 20.6 mrad. The mechanism is implicit identification from the history, as the agenda anticipated.</li>
<li><strong>Gradients through the plant, if tamed.</strong> Backpropagation through MuJoCo's finite-difference derivatives works once the chained Jacobians through contact are clipped per sample at a multiple of the batch median, the horizon is short with a critic bootstrap (gamma 0.995, ramped in), the tracking kernels are bounded (20 mrad) and the torque-rate penalty is light (0.003; at 0.03 it dominated the objective and the residual gained by softening). With those four elements the residual beats the tuned PD on every family and group of protocol v2 (D3) and is the base of every later student.</li>
<li><strong>Specify the contact behaviour, then imitate it.</strong> The wanted response was written as a law, a force-bounded spring on the actual error with stop-and-wait when blocked, executed by a privileged oracle, validated against the metrics, and distilled by DAgger in ten minutes per seed: blocked excess force 57 to 15 N on the sphere plant, and on the gripper 5 N against 23 for the best classical and 43 for the student's own base, apparent stiffness 1440 to 390 N/m, and a contact stiffness near the teacher's 500 N/m on held-out compliant surfaces where the PD keeps its own 1 kN/m. Five rounds of reinforcement learning never found this behaviour by exploration and did not improve it once found.</li>
<li><strong>The external torque is the cue.</strong> The link-side torque minus the model's prediction detects contact within 50 ms for 93 to 97% of onsets (error and velocity cues: 61 to 78%), and separates contact from a payload once payload-carrying free motion is in the fit set. Without it the features of the residual carried contact only through the accumulated error, 0.7 s late.</li>
<li><strong>Protect the base during distillation.</strong> With the teacher driving only once and free-space decisions weighted three times, the student keeps its base's free space within 3%; the first recipe (the teacher driving in three iterations, unweighted labels) gave up 20 to 40% of it.</li>
<li><strong>Give the learned controller the model term.</strong> The nominal inverse-dynamics feedforward in the PD base, a free-space-only base and selection on free space took the student from 11.9 to 6.9 mrad; the student reproduces its base, so what remained was the base's reward.</li>
<li><strong>Ideal-robot labels as the free-space teacher, referenced to the nominal law.</strong> Supervised labels that make the sampled plant reproduce the nominal robot's response halve the feedforward PD's error (2.6 mrad) and hold at 1.6 against 8.1, in every seed, provided the label is taken relative to the nominal law rather than the student's own command (item 6 of Section 5).</li>
<li><strong>Keep the 1 kHz loop under the learned part.</strong> The residual form rides on the PD between decisions and under latency: the students move by at most 5% at 5 ms. The same history, same training, same teachers with a direct torque output lose every free-space family (Section 5, item 2).</li>
<li><strong>Contact behaviour transfers across plants.</strong> The student trained on the sphere plant ran unchanged on the gripper plant and kept its contact behaviour (2.3 N above the cap on blocked targets); its free-space edge did not transfer and was restored by retraining.</li>
</ol>"""

FAILED = """<ol>
<li><strong>Training through the penalty-contact torch model (M2).</strong> Validation in the plant worsened whenever the contact stage entered training while the in-model loss fell: exploitation of the contact approximation. Superseded by the plant as the medium; free-space training through the model was fine, and a learned model matched it (M4).</li>
<li><strong>Direct torque networks.</strong> The worst parameterisation through the known model (M2), untrainable by PPO (M5), 136 to 141 N blocked excess at 1 kHz (D2), collapsed at the contact stage under finite-difference gradients (D3), and in round 2 the best hold tracker as a base yet the worst student: its torque is open-loop for the 10 ms it is held, it chatters at 1953 Nm/s, and a 5 ms latency degrades it threefold. One architecture and recipe were tried; the result is about this recipe.</li>
<li><strong>Reinforcement learning, five rounds.</strong> PPO was worse than backpropagation for every parameterisation under protocol v1 (M5); in the plant it reached parity with its warm start and cut forces by a flinch at the reward's soft limit (D4); under the cap reward it pushed less but stayed stiff and never explored the gain head (D5); it neither improved the distilled students nor produced stop-and-wait from the D5 arm (D6). The training return rose in every round while the protocol's contact composites did not follow. The diagnosis from saved episodes: yielding at a blocked surface is a large, coordinated residual torque that no optimiser found by exploration, and on sliding references the reward preferred intermittent light contact. Budget per round was small (150 to 300 updates); the pattern was the same in each.</li>
<li><strong>Force penalties as reward terms.</strong> Instantaneous terms made the contact-stage gradients three to four times larger and noisier and trained worse than the no-force twins (D2); filtered terms attached to the gradient cost free-space tracking three times in a row (D5). Detached from the gradient they were harmless and taught the features to sense contact, but they did not produce compliance.</li>
<li><strong>Selection on the cap composite for a base.</strong> The actor-critic residual's best checkpoint under the cap composite was a contact-stage iteration whose free space was half as good as its stage-1 optimum (M10). A selection rule changes what a method is; the base is now selected on free space and the contact behaviour is the student's.</li>
<li><strong>Ideal-robot labels under the student's own command.</strong> Labelling the torque the plant needs under the student's full command lets the target drift with the student's residual: every DAgger iteration degraded hold and free motion (M10's ideal-robot run; the first two-teacher form). Referenced to the nominal law the same labels are the best free-space teacher.</li>
<li><strong>Learned gains without a residual, phase-scheduled gains, the target preview.</strong> Gain schedules alone went soft everywhere (M2); hand-scheduled gains spike torque at switches and sag under a payload (M6); a preview of future targets adds nothing beyond the classical feedforward (D6). The bounded gain multipliers survive only as a head next to the residual.</li>
<li><strong>Small validation sets.</strong> Two cases per family misranked the contact arms of D4 against the 16-case test set; eight cases per contact family (two to four per free-space family) are used since, and no ranking is claimed from training-time validation.</li>
</ol>"""

GAPS = """<ol>
<li><strong>Policy-like targets, the one miss.</strong> At 1.05 to 1.06 of the best classical in every group, with better tracking and a higher torque rate. The ideal-robot labels are dropped where the nominal law saturates, which is exactly at target jumps, so the student's behaviour there is the base's; nothing in the labels asks for smoothness.</li>
<li><strong>The press band against the cap.</strong> The 7 N floor puts pressing inside the 7 to 12 N band (10.6 N mean) at 5 N above the 15 N cap on the sliding families, where light contact at 9 to 15 N detaches 20 to 70 times per slide for every controller on both plants (report 14). The contact-loss metric on sliding families measures the simulator as much as the controller; the plant question comes before further tuning of the teacher.</li>
<li><strong>Task success on sixteen cases.</strong> Push success 52% against 31% is eight or nine boxes against five; box yaw is the failure for every controller and is a task-policy problem. Placement is at the ceiling for everyone, so the task half of the criterion is met on one task and tied on the other.</li>
<li><strong>Privileged supervision.</strong> Both teachers use what only a simulator gives: the oracle knows the surface and the payload; the ideal-robot label is the sampled plant's inverse dynamics. On hardware the oracle's role could be taken by a force-sensed or estimated impedance law, and the ideal-robot label needs a system identification or a learned plant model (Milestone 4's model is the candidate). Neither path has been built.</li>
<li><strong>Contact sensing stays weak in the features.</strong> Linear probes decode contact from the trunk features at 0.71 to 0.76 balanced accuracy for every controller, students included, although the behaviour is right; the external-torque input carries the cue, the features do not represent it explicitly.</li>
<li><strong>The classical bar is modest.</strong> Forty CMA-ES iterations on four cases per family, one tuning seed, a feedforward from the nominal model without instance adaptation. A better-tuned classical set, or an adaptive classical law, would raise the bar the learned controller is measured against.</li>
<li><strong>The ceiling is the teacher.</strong> The student reaches the oracle's contact composites; whether a better contact law exists is not tested, since the oracle's parameters (500 N/m, 0.7 damping ratio, 200 ms blend) were set once from hardware practice.</li>
<li><strong>Smoothness and error bursts.</strong> The student's torque rate in free motion is above the stiff PD's (159 against 115 Nm/s) and higher in contact; it is within the protocol's weight but not a win. In free motion the student also shows transient error bursts on single joints in a few cases (peaks of 24 to 47 mrad across seeds where the feedforward PD peaks at 6), which cost it the cases where the nominal model is already accurate (Section 2.4, free-motion clip); the ideal-robot labels teach the model correction, not a bound on the correction's rate.</li>
<li><strong>Not attempted.</strong> The real robot (Milestone 7, unauthorised), a second robot and task conditioning (Milestone 8), the representation study, and every claim rests on three seeds and sixteen test cases per family.</li>
</ol>"""

SCALING = """<p>The recipe was developed on a laptop with runs of under an hour. The following items were set once, at that budget, and are where more data, iterations and compute should go first. Each is a question with a measurable answer under the existing protocol.</p>
<ol>
<li><strong>Replication before anything else.</strong> Five or more seeds per method, 64 or more test cases per family and per task, and three tuning seeds of the classical set with a larger CMA-ES budget. The paired intervals of the present wins are the evidence to read; several contact-family wins are large, the task win and the policy-like miss are not.</li>
<li><strong>The base's budget and reward (item E, not run).</strong> 250 iterations of 24 windows were enough for the base to reach 6.9 mrad; the kernel half-width (20 mrad) and the torque-rate weight (0.003) were chosen at that budget and do not distinguish 5 from 7 mrad. With more iterations and batch, re-tune both and test whether the base alone closes the free-motion gap, which would make the ideal-robot teacher unnecessary.</li>
<li><strong>Distillation budget.</strong> Twelve iterations of 48 episodes with a 300 thousand-row cap, four epochs per iteration, a teacher-driving schedule of one iteration and a free-space weight of 3, all set once. Scale the rows with the data, lengthen the schedule, and add a smoothness term to the free-space labels (or the oracle's labels on the policy-like family) for the one miss.</li>
<li><strong>The training distribution.</strong> Instances and payloads are sampled inside the evaluation ranges by design; compliant surfaces, the task sequence and the held-out workspace are not trained on. With more compute, widen the surface geometries, speeds and payloads of training while keeping at least one group held out per axis so the generalisation claims survive.</li>
<li><strong>Validation.</strong> Two to four free-space and eight contact cases per family select checkpoints; every checkpoint is kept. With compute, validate on more cases and report the selection rule with the alternative checkpoint, since selection changed what two methods looked like (Section 5, item 5).</li>
<li><strong>The learned model as the medium.</strong> Learned-model gradients matched the known model (M4) and were stable through contact at a third of the cost (D3); on a cluster the finite-difference gradients, whose cost grows with the state dimension, are the expensive part of stage 1. The learned model is also the hardware path of the ideal-robot label (a learned plant model in place of the sampled plant's inverse dynamics).</li>
<li><strong>The teacher.</strong> Sweep the oracle's stiffness, floor and blend against the band and the cap, and settle the plant's light-contact detachment (a finite contact patch or a different pad model) before the sliding families are used as evidence.</li>
<li><strong>Architecture, untouched since ADR 0005 and 0007.</strong> History length and stride, width 128 and two layers, the 10 ms decision period, the output bounds, an output low-pass and Fourier features. None was varied on plant v3; a representation study is cheap next to training and was deferred twice.</li>
<li><strong>Beyond the protocol.</strong> More tasks and task conditioning (Milestone 8), a second robot with its own model and controller, and the real robot once authorised, where the inputs and outputs of the controller are already those a Panda exposes and the open question is the supervision (item 4 of Section 6).</li>
</ol>"""

POINTERS = """<ul>
<li><strong>Project documents (private; not part of the public release).</strong> {research} (the agenda, approved milestones and the integrity rules), {context} (scientific state, blockers, pending proposals), {adr} (ten decision records; ADR 0010 holds the criterion and the round-2 amendment), {protocol} (families, groups, metrics, composites); the metrics and composites are also defined in <code>llc/eval/metrics.py</code>.</li>
<li><strong>Reports.</strong> {r15}, {r17} and {r16} for the final evidence; {r13} for the contact law and the cue audit; {r09} for the training recipe of the base; {r14} for the plant and the task families.</li>
<li><strong>Code.</strong> Controllers in <code>llc/controllers/</code> (PD, feedforward PD, the oracle, the learned controller with its latency queue); the policy, history buffer, actor-critic, distillation and ideal-robot labels in <code>llc/learning/</code>; references, cases, metrics and the parallel evaluation in <code>llc/eval/</code> and <code>llc/references.py</code>; the two plants, instances, gripper and tasks in <code>llc/sim/</code>. Training entry points: <code>experiments/m1_tune_gains.py</code> (gains), <code>experiments/d2_train.py --feedforward --val-objective free</code> (stage 1), <code>experiments/d7_distill.py --free-teacher ideal</code> (stage 2); the benchmark <code>experiments/d_evaluate.py</code> and the report builders <code>experiments/d_report.py</code>, <code>experiments/checkpoint_2.py</code>, <code>experiments/handoff.py</code>. The launchers of the final runs are tracked under <code>runs/m9/</code>, <code>runs/m10/</code> and <code>runs/m10r2/</code>; the checkpoints, metrics, logs and episodes under <code>runs/</code> are not under version control.</li>
<li><strong>Environment.</strong> <code>uv</code> manages the dependencies; run with <code>PYTHONPATH=. uv run --no-sync</code> and <code>LLC_PLANT=v3</code>; the notebook is built by <code>research/build.py</code> from <code>research/reports/</code> and <code>research/catalog.json</code>; 65 tests in <code>tests/</code>.</li>
</ul>"""


def videos_section() -> str:
    """The ten representative clips rendered by experiments/handoff_videos.py, if present."""
    videos = REPORT / "data" / "videos.html"
    if not videos.exists():
        return ""
    intro = """<h3>2.4 Representative episodes</h3>
<p>Ten side-by-side replays, one per behaviour the protocol asks for: the best classical controller of the family on the left, the learned controller (seed 0) on the right, the privileged oracle between them in the contact clips. Every view replays the stored joint trajectory of the same test case through the same scene and camera; the gray sphere is the commanded tool position; the plot strip below each clip shows the full episode with a cursor at the displayed time, and the joint RMSE and peak force under each view accumulate from the start. Six of the in-distribution clips use the benchmark's representative case per family (the median composite of the tracking-tuned PD); the free-motion clip shows an instance on which the nominal model is wrong; the held-out and task clips state their selection rule. What to look for: in free space the learned controller's error curve lies below the PD's with little visible difference in the motion (the per-case losses are counted in Section 2.3); in contact the force curve stays bounded where the PD's does not; in the task clips the object's motion follows from the force the controller lets through.</p>"""
    return intro + "\n" + videos.read_text()


# ------------------------------------------------------------------------------------------ page
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/m10r2/metrics_test.json")
    args = ap.parse_args()
    _data, groups = load(Path(args.metrics))
    for g in groups.values():
        add_best_classical(g)
    n = headline_numbers(groups)
    setting = SETTING.format(agenda=page("research", "research agenda"), protocol=protocol_table())
    pointers = POINTERS.format(research=page("research", "Research agenda"), context=page("context", "Current context"), adr=page("adr", "Decisions"), protocol='<a href="../protocols/evaluation-v3/index.html">Protocol v3</a>',
                               r15=rep("15-m10-across-the-board", "report 15"), r17=rep("17-m10-round-2", "report 17"), r16=rep("16-checkpoint-2", "Checkpoint 2"), r13=rep("13-dd-impedance", "report 13"),
                               r09=rep("09-dd-gradients", "report 09"), r14=rep("14-m9-plant-v3", "report 14"))
    parts = ["<h2>Summary</h2>\n" + summary(n),
             "<h2>1. The problem and the setting</h2>\n" + setting,
             "<h2>2. The controller that resulted</h2>\n" + architecture_figure() + RECIPE + headline_table(n) + fig_criterion_grid(groups) + VERDICT + verdict_single(n) + videos_section(),
             "<h2>3. The hypothesis space</h2>\n" + matrix_table() + OTHER_AXES + "<h3>3.1 Chronology</h3>\n" + chronology_table(),
             "<h2>4. What worked, and why</h2>\n" + WORKED,
             "<h2>5. What did not work, and why</h2>\n<p>Each item is an investigated negative result unless marked budget-limited; the mechanism is stated where it was established from saved episodes, reward decompositions or probes.</p>\n" + FAILED,
             "<h2>6. Where the gaps are</h2>\n" + GAPS,
             "<h2>7. What to revisit when scaling up</h2>\n" + SCALING,
             "<h2>8. What a run costs</h2>\n" + cost_table(),
             "<h2>9. Pointers</h2>\n" + pointers]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report.html").write_text("\n\n".join(parts))
    print("wrote", REPORT / "report.html")
    for k, v in sorted(n.items()):
        if isinstance(v, float):
            print(f"  {k:22s} {v:9.3f}")
    for k, st in n["single"].items():
        print(f"  {k:28s} within 5% of the per-family best {st['within']}/{st['total']}, mean ratio {st['ratio']:.2f}, student met {st['student_met']}/{st['total']}")


if __name__ == "__main__":
    main()
