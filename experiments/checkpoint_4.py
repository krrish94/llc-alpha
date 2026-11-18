"""Checkpoint 4 (Milestone 12, ADR 0012): a standalone summary of the consolidation on the torque-interface Panda, in the style
of Checkpoints 1 to 3: the question, the controllers, the task set and the plant distribution, the evidence of report 23, the
deployment bundle, the lessons, and what to take to hardware.

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/checkpoint_4.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, method_color, reference_color  # noqa: E402
from report_api import figure, table  # noqa: E402

from experiments.d7_extras import label  # noqa: E402
from experiments.m10_extras import criterion_table  # noqa: E402
from experiments.m12_extras import (STUDENTS, criterion_counts, deployment_table, depth_table, families_table, load_metrics, scale_table, task_table,  # noqa: E402
                                    with_reference)

REPORT = Path("research/reports/24-checkpoint-4")
FIG = REPORT / "figures"


def _color(key: str) -> str:
    try:
        return method_color(key)
    except KeyError:
        return method_color("distill_v3_r2_two_teachers")


def fig_hero(entries: list[tuple[str, int, int, str]]) -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 0.6 + 0.42 * len(entries)))
    y = np.arange(len(entries))[::-1]
    for yy, (lab, ok, tot, key) in zip(y, entries):
        frac = ok / tot if tot else 0.0
        ax.barh(yy, 100 * frac, color=_color(key), height=0.65)
        ax.text(100 * frac + 1.5, yy, f"{ok} of {tot}", va="center", fontsize=8)
    ax.axvline(100, color=reference_color(), lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([e[0] for e in entries], fontsize=8)
    ax.set_xlim(0, 118); ax.set_xlabel("family-and-group rows within 5% of the reference, or better (%)")
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG / "hero.svg", bbox_inches="tight"); plt.close(fig)
    return figure("figures/hero.svg", "The across-the-board criterion on protocol v5 (fourteen families, every group, 32 cases per family): rows on which a student is within 5% of the reference, or better; against the best global classical controller (the gate) and against the per-family PD (the goal), one seed each.", "fig-hero", wide=True)


def main() -> None:
    path = Path("runs/m12/metrics_test.json")
    data, groups = load_metrics(path)
    learned = [k for k in STUDENTS if k in groups["in_distribution"]]
    fam_groups = with_reference(groups, ("pd_family",)) if "pd_family" in groups["in_distribution"] else {}
    cg, cf = criterion_counts(groups, learned), (criterion_counts(fam_groups, learned) if fam_groups else {})
    entries = [(f"{label(k)}, against the global classical set", cg[k][0], cg[k][1], k) for k in learned] + [(f"{label(k)}, against the per-family PD", cf[k][0], cf[k][1], k) for k in learned if k in cf]
    hero = fig_hero(entries)
    methods = [k for k in ("best_classical", "pd_family", "oracle_pd") + tuple(learned) if k in groups["in_distribution"]]
    videos = Path("research/reports/23-m12-consolidation/data/videos.html")
    clips = ""
    if videos.exists():
        src_media = Path("research/reports/23-m12-consolidation/media")
        (REPORT / "media").mkdir(parents=True, exist_ok=True)
        for f in src_media.glob("*"):
            shutil.copy2(f, REPORT / "media" / f.name)
        clips = "<h2>7. The new families in motion</h2>\n" + videos.read_text()
    export = Path("runs/m12/export_l4/spec.json")
    export_section = ""
    if export.exists():
        spec, timing = json.load(open(export)), json.load(open(export.parent / "timing.json"))
        rows = [["Parameters", f"{spec['policy']['parameters']:,}"], ["Observation", f"{spec['policy']['obs_dim']} values: " + "; ".join(f"{o['name']} ({o['size']})" for o in spec["observation"])],
                ["Decision period", f"{spec['timing']['decision_period_ms']:.0f} ms, the held law at 1 kHz"], ["Inference time per decision (one thread, TorchScript)", f"{timing['torchscript_ms_per_decision']:.3f} ms"],
                ["Base controller", f"PD with the nominal inverse-dynamics feedforward, kp {np.round(spec['base']['kp'], 0).tolist()}, kd {np.round(spec['base']['kd'], 1).tolist()}"],
                ["Inputs on hardware", "; ".join(spec["inputs_on_hardware"])], ["Unknown to the policy", ", ".join(spec["unknown_to_the_policy"])]]
        export_section = "<h2>8. The deployment bundle</h2>\n<p>What a Panda needs to run the 4-layer student (experiments/m12_export.py): the policy as TorchScript, the observation layout, the base law and the timing.</p>\n" + table(["Item", "Value"], rows, caption="The exported bundle runs/m12/export_l4.")
    parts = [f"<h2>Summary</h2>\n{SUMMARY}", "<h2>1. The question</h2>\n" + QUESTION, "<h2>2. Controllers</h2>\n" + CONTROLLERS, "<h2>3. The task set and the plant distribution</h2>\n" + TASKS,
             "<h2>4. The evidence</h2>\n" + hero + criterion_table(groups, learned, caption="Against the best global classical controller of each family (the gate of ADR 0012).")
             + (criterion_table(fam_groups, learned, caption="Against the per-family PD (the goal of ADR 0012).") if fam_groups else "") + families_table(groups, methods) + task_table(groups, methods),
             "<h2>5. Depth and data scale</h2>\n" + depth_table(data, groups) + scale_table(groups), "<h2>6. Deployment groups</h2>\n" + deployment_table(groups, learned)]
    if clips:
        parts.append(clips)
    if export_section:
        parts.append(export_section)
    parts += ["<h2>9. Lessons</h2>\n" + LESSONS, "<h2>10. Next</h2>\n" + NEXT]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report.html").write_text("\n\n".join(parts))
    print("wrote", REPORT / "report.html", "| criterion:", {k: (cg[k], cf.get(k)) for k in learned})


SUMMARY = """<p><strong>Verdict.</strong> RESULTS_PENDING</p>"""
QUESTION = """<p>Does the learned low-level controller of Checkpoints 2 and 3, trained on ten times the data under a plant distribution wide enough to cover a real Panda, hold its across-the-board result on a task set extended with contact-rich and sustained-contact work, and does depth buy anything there? The controller that comes out is the candidate for hardware; hardware itself stays unauthorised.</p>"""
CONTROLLERS = """<p>The global PD (the tracking-tuned and cap-tuned sets of Milestone 10, feedforward form), the per-family PD (a gain set tuned on each family alone by CMA-ES on the cap composite, the strongest fixed-gain reference), the privileged oracle (its plane per task: the table, the wall, the held block's contact plane, the hole's top plane with a weak lateral spring while the peg is blocked, the panels' stops), and the two-phase students: a residual base trained through the plant on free space (Milestone 10 round 2's recipe) at 4 layers of 256 and at 10 layers of 256 with residual blocks and layer normalisation, then DAgger from the ideal-robot labels in free space and the oracle in contact for twenty iterations of 320 episodes with no cap on the label store, the teacher's share 1, 0.5, 0.25 then 0, validation on 96 cases over every family. The scale control is the 4-layer student at the Milestone 10 budget. One seed each.</p>"""
TASKS = """<p>Fourteen families, 32 cases each: the nine of protocols v3 and v4 (hold, free motion, policy-like targets, steps, press-and-slide, blocked, task, push, pick with a grasp) and five new ones: wiping under the force band, erasing with a held block, peg-in-hole with a chamfer (0.75 or 1.5 mm of clearance per side, the reference 2 to 6 mm off), a drawer pushed to its stop along a drifting reference, a door pushed closed along a straight line. The training distribution: robot instance (masses ±15%, inertias ±20%, centres of mass ±2 cm, damping and armature ×0.5 to 2, actuator gain ±10%, friction to 0.5 Nm), payload 0 to 3 kg unknown to the student, sensor noise 0.1 to 0.5 mrad, compute latency 0 to 10 ms, actuation delay 0 to 5 ms, actuator lag 0 to 3 ms, the base PD's stiffness ×0.3 to 3 and damping ×0.5 to 2 (given to the student as inputs, with the latency), surfaces rigid, 2 kN/m or 500 N/m with friction 0.3 to 1.0, object masses ±50%. Evaluation groups: those of v3 and v4, the base at half and double stiffness, a 3 ms actuator lag, compute latencies of 5 and 10 ms.</p>"""
LESSONS = """<ul>
<li>RESULTS_PENDING</li>
</ul>"""
NEXT = """<ul>
<li>Seeds 1 and 2 of the recipe that comes out; the Cartesian impedance base and the object-along-a-wall family, deferred by the advisor; hardware when authorised, with the deployment bundle.</li>
</ul>"""

if __name__ == "__main__":
    main()
