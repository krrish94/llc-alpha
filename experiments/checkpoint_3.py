"""Checkpoint 3 (Milestone 11, ADR 0011): a standalone summary of the learned controller on position-controlled arms, in the
style of Checkpoints 1 and 2: the question, the interface model, the protocol, the evidence of reports 19 to 21 and the
continuation (amendment 3), the corrections, the lessons, and what to take to hardware or to scale.

    LLC_PLANT=v3 PYTHONPATH=. uv run --no-sync python experiments/checkpoint_3.py
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

from experiments.d7_extras import load, mean_metric  # noqa: E402
from experiments.d_report import add_best_classical  # noqa: E402
from experiments.m10_extras import criterion_table  # noqa: E402
from experiments.m11_extras import (PROFILE_LABELS, interface_table, load_profiles, motor_table, probe_table, profile_table, retune_table, stiffness_table, torque_table,  # noqa: E402
                                    transient_table, variants_table, velfilter_table)

REPORT = Path("research/reports/22-checkpoint-3")
FIG = REPORT / "figures"
RUNS = Path("runs/m11")


def criterion_count(path: Path, key: str, tolerance: float = 0.05) -> tuple[int, int]:
    """Rows within the tolerance of the best classical controller, over every group and family of a metrics file."""
    if not path.exists():
        return 0, 0
    _, groups = load(path)
    ok = tot = 0
    for gname, g in groups.items():
        if gname.startswith("latency") or key not in g:
            continue
        add_best_classical(g)
        for r in sorted({m["regime"] for m in g[key]}):
            v, b = mean_metric(g[key], r, "composite_cap"), mean_metric(g["best_classical"], r, "composite_cap")
            if np.isfinite(v) and np.isfinite(b):
                tot += 1; ok += int(v <= (1 + tolerance) * b)
    return ok, tot


def _color(key: str) -> str:
    """The palette's colour for a method key; the Milestone 11 students share the two-teacher student's colour."""
    try:
        return method_color(key)
    except KeyError:
        return method_color("distill_v3_r2_two_teachers" if key.startswith("distill_host") else "distill_v3_transfer")


def fig_hero(entries: list[tuple[str, int, int, str]]) -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 0.5 + 0.42 * len(entries)))
    y = np.arange(len(entries))[::-1]
    for yy, (lab, ok, tot, key) in zip(y, entries):
        frac = ok / tot if tot else 0.0
        ax.barh(yy, 100 * frac, color=_color(key), height=0.65)
        ax.text(100 * frac + 1.5, yy, f"{ok} of {tot}", va="center", fontsize=8)
    ax.axvline(100, color=reference_color(), lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([e[0] for e in entries], fontsize=8)
    ax.set_xlim(0, 118); ax.set_xlabel("family-and-group rows within 5% of the best classical controller, or better (%)")
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG / "hero.svg", bbox_inches="tight"); plt.close(fig)
    return figure("figures/hero.svg", "The across-the-board criterion (ADR 0010, through the interface in ADR 0011): rows of the family-and-group table on which a learned controller is within 5% of the best classical controller of that family, or better. The torque-interface student of Milestone 10 for reference; the host-rate students of Milestone 11 against the classical set through the same profile, the re-tuned PD included where it was evaluated; the YAM student through its own interface, with and without the randomised DC-motor layer.", "fig-hero", wide=True)


def profiles_table() -> str:
    rows = [["P0", "the torque interface of reports 01 to 18", "1 ms", "0", "torque", "exact"],
            ["P1", "MIT mode on CAN (DM4310/DM4340; the YAM)", "4 ms", "1 ms", "position, velocity, Kp, Kd, feedforward torque per frame (16, 12, 12, 12, 12 bits)", "position 16 bit per 25 rad, velocity 12 bit per 60 rad/s"],
            ["P2", "Dynamixel X position mode (Koch, GELLO class)", "10 ms", "2 ms", "goal position (4096 per turn), P and D gains in RAM, a current cap", "position 1.53 mrad, velocity 24 mrad/s, current 2.69 mA"],
            ["P3", "Feetech STS position mode (SO-101)", "20 ms", "3 ms", "goal position (4096 per turn), a torque limit; gains fixed", "position 1.53 mrad, velocity 77 mrad/s"]]
    return table(["Profile", "Modelled after", "Host period", "Bus delay", "Command channels", "Feedback"], rows, caption="The interface profiles (ADR 0011, decision 1). The servo law runs at 1 kHz on the quantised measurement with the frame held between host instants; the model-based interface folds the channels a profile lacks into the goal, with the reference advanced over the bus delay and half the hold.", css_class="wide")


def main() -> None:
    profiles = load_profiles()
    yam_path, yam_motor = Path("runs/m11/yam/metrics_p1_student.json"), Path("runs/m11/yam/metrics_p1_motor.json")
    entries = []
    m10 = Path("runs/m10r2/metrics_test.json")
    if m10.exists():
        entries.append(("Milestone 10 student, torque interface (Panda)", *criterion_count(m10, "distill_v3_r2_two_teachers"), "distill_v3_r2_two_teachers"))
    for p in ("p1", "p2", "p3"):
        f = next((Path(x) for x in (f"runs/m11/metrics_phase2d_{p}.json", f"runs/m11/metrics_phase2c_{p}.json", f"runs/m11/metrics_students_{p}.json") if Path(x).exists()), None)
        if f is not None:
            entries.append((f"Host-rate student through {PROFILE_LABELS[p]} (Panda, recipe)", *criterion_count(f, f"distill_host_{p}"), f"distill_host_{p}"))
    for key, lab in (("distill_host_p1_f1", "P1 student with the force-space label term (Panda)"), ("distill_host_p1_beta", "P1 student, the teacher's share kept for three more iterations, seeds pooled (Panda)"), ("distill_host_p1_beta_l8", "The same with eight layers (Panda)")):
        for f in (Path("runs/m11/metrics_phase2e_p1.json"), Path("runs/m11/metrics_phase2d_p1.json"), Path("runs/m11/metrics_phase2c_p1.json")):
            if f.exists() and key in load(f)[1].get("in_distribution", {}):
                entries.append((lab, *criterion_count(f, key), key)); break
    if yam_path.exists():
        entries.append(("YAM student through P1 (its own interface)", *criterion_count(yam_path, "distill_yam_p1"), "distill_yam_p1"))
    if yam_motor.exists():
        entries.append(("YAM student through P1 under the randomised DC-motor layer", *criterion_count(yam_motor, "distill_yam_p1"), "distill_yam_p1"))
    hero = fig_hero(entries) if entries else ""
    sweep_src = Path("research/reports/19-m11-interfaces/figures/sweep.svg")
    sweep = ""
    if sweep_src.exists():
        FIG.mkdir(parents=True, exist_ok=True); shutil.copy2(sweep_src, FIG / "sweep.svg")
        sweep = figure("figures/sweep.svg", "Host period and bus delay (the P2 structure, the feedforward PD): the composites per family over the period at a 2 ms delay and over the delay at a 10 ms period. The hold matters beyond 20 ms, the delay hardly up to 10 ms.", "fig-sweep", wide=True)
    yam_crit = ""
    if yam_path.exists():
        _, yg = load(yam_path)
        for g in yg.values():
            add_best_classical(g)
        yam_crit = criterion_table(yg, ["distill_yam_p1"], caption="The YAM through P1: the student against the best classical controller of each family and group (the shipped gains with and without the feedforward, the re-tuned PD), ADR 0011 criterion (iv), one seed.")
    panda_crit = ""
    f21 = next((Path(x) for x in ("runs/m11/metrics_phase2e_p1.json", "runs/m11/metrics_phase2d_p1.json", "runs/m11/metrics_phase2c_p1.json") if Path(x).exists()), None)
    if f21 is not None:
        _, pg = load(f21)
        for g in pg.values():
            add_best_classical(g)
        keys = [k for k in ("distill_host_p1", "distill_host_p1_beta", "distill_host_p1_beta_l8", "distill_host_p1_f1") if k in pg["in_distribution"]]
        panda_crit = criterion_table(pg, keys, caption="The Panda through P1: the host-rate students against the best classical controller through the same profile (the two tuned PDs, their feedforward variants and the PD re-tuned through the profile).")
    parts = [f"<h2>Summary</h2>\n{SUMMARY}",
             "<h2>1. The question</h2>\n" + QUESTION,
             "<h2>2. The interface model</h2>\n" + INTERFACE + profiles_table(),
             "<h2>3. Protocol v4</h2>\n" + PROTOCOL,
             "<h2>4. The classical set through the profiles (Panda)</h2>\n" + ABLATION + (profile_table(profiles) if profiles else "") + (torque_table(profiles) if profiles else "") + sweep + (stiffness_table(profiles) if profiles and "p0" in profiles else "") + retune_table() + velfilter_table(),
             "<h2>5. The learned interface</h2>\n" + LEARNED_IFACE + interface_table(),
             "<h2>6. The host-rate students (Panda)</h2>\n" + hero + STUDENTS + panda_crit + BLOCKED + transient_table() + variants_table([RUNS / "metrics_phase2c_p1.json", RUNS / "metrics_phase2d_p1.json", RUNS / "metrics_phase2e_p1.json"]),
             "<h2>7. The YAM</h2>\n" + YAM + yam_crit + motor_table(yam_path, yam_motor, ["pd_feedforward", "pd_shipped_feedforward", "oracle_pd", "distill_yam_p1"]) + transient_table(Path("runs/m11/yam/blocked_transient_p1.json"), cap=12.0),
             "<h2>8. What the representations encode</h2>\n" + PROBES + probe_table(RUNS / "probe_ext_student_p1.json", extended=True, caption="The Panda P1 student: the extended probing targets, from the observation and from every trunk layer.") + probe_table(Path("runs/m11/yam/probe_ext_student_p1.json"), extended=True, caption="The YAM student: the same targets."),
             "<h2>9. Corrections made during the milestone</h2>\n" + CORRECTIONS,
             "<h2>10. Lessons</h2>\n" + LESSONS,
             "<h2>11. Next</h2>\n" + NEXT]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report.html").write_text("\n\n".join(parts))
    print("wrote", REPORT / "report.html", "| criterion entries:", [(e[0][:40], e[1], e[2]) for e in entries])


SUMMARY = """<p><strong>Verdict.</strong> The learned low-level controller of Checkpoint 2 carries over to position-controlled arms as a host-rate student: a network that decides once per host frame on the servo's quantised feedback and sends an impedance frame, distilled from scratch through the interface by the two-teacher recipe. On the i2rt YAM through its own interface (MIT mode on CAN at 250 Hz) the student meets the across-the-board criterion on 32 of 33 family-and-group rows against the best classical controller through the same interface, in one seed, and keeps every row under a DC-motor layer whose unknowns are randomised per case. On the Panda, where the same profiles are an ablation of the interface alone, the host-rate students meet 24 to 26 of 33 rows through the three position profiles against the torque-interface classical set (18 to 22 once the PD re-tuned through the profile joins it), winning hold, steps, the task family and the pick and losing the blocked family and the fast free-space families through their torque rate; keeping the teacher in the loop for three more iterations takes the blocked family in two of three seeds (23 of 33 rows pooled, through P1). A learned interface below the controller finds nothing under the model-based one: with the reference advanced over the bus delay and half the hold, the classical mapping is at the floor the servo's own loop sets. The torque rate every controller pays through a position interface is the interface's own, the held setpoint's staircase on a quantisation floor, not an assumption about the servo's velocity estimate that hardware would relax.</p>
<p><strong>What changed since Checkpoint 2.</strong> A servo layer with four interface profiles and a model-based mapping; protocol v4 (a torque-tracking family, the pick with a grasp); the plant generalised to a second robot (the YAM) with the protocol's force scale set per robot; host-rate students at the host period with deeper trunks; a probing study with contact-geometry targets; a DC-motor layer under the servo law; and the blocked family at the host rate on the Panda diagnosed to the impact transient under payload, with the recipe varied in label space (section 6).</p>"""

QUESTION = """<p>Most arms a laboratory can buy are position-controlled: the host sends a goal, or an impedance frame, over a bus at 50 to 250 Hz, and a servo loop inside the motor holds it. The controller of Checkpoint 2 commands torque at 1 kHz. Milestone 11 asks what of it survives when the torque abstraction has to pass through such an interface, and what a learned controller should be on an arm of that kind: a learned interface that makes the servo behave like a torque source, the controller itself rebuilt at the host rate, or both. The advisor's framing: one controller per robot type, the learned controller as the torque-command interface, hardware not yet allowed, so everything is in simulation with the interface modelled from the drivers and the datasheets, and identified where public data allows.</p>"""

INTERFACE = """<p>The plant takes a servo profile: how often the host may command, with what channels and resolution, how long the bus takes, and what the servo holds between frames. Every controller of the project exposes an impedance frame (goal position and velocity, gains, feedforward torque, cap); a torque command is the frame with zero gains, held with the servo's own spring. The model-based interface (the classical baseline) maps the frame to the profile's channels so that the servo's torque equals the commanded law at the frame's arrival: the reference advanced over the bus delay and half the hold (zero-order-hold compensation), the measured state predicted over the delay, the channels the profile lacks folded into the goal. The servo law runs at 1 kHz on the quantised measurement. On the Panda the servo's loop is abstracted as the tuned PD in torque units, so that the profiles change the interface and nothing else; on the YAM a DC-motor layer with randomised unknowns sits under the law for the robustness check.</p>"""

PROTOCOL = """<p>Protocol v4 adds to v3 a torque-tracking family (holds, steps, chirps up to the host Nyquist rate, pure-torque chirps on the upper command; metrics: realised-torque error as a fraction of the stall torque, delay, bandwidth, saturation, torque rate) and the pick with a grasp: approach, descend over the block at a fixed opening, grasp (the weld activates only if the block's centre is inside the gap at the grasp instant), lift, carry, place; grasp success, the block's disturbance before the grasp and the place metrics. The force scale is set per robot (Panda cap 15 N, band 7 to 12 N; YAM cap 12 N, band 5 to 8 N); the families, splits, held-out groups and composites are those of v3, and the criterion is ADR 0010's through the interface: within 5% of the best classical controller of each family through the same profile, or better.</p>"""

ABLATION = """<p>The same classical controllers through the four profiles. Tracking and contact behaviour survive every profile (the realised contact stiffness within 3% of the torque interface's); smoothness does not (free-motion torque rate 115, 587, 1074 and 1710 Nm/s from P0 to P3). The torque-tracking family separates the profiles the way the bus does (0.4 to 1.3% of the stall torque at P1 with delays of 1 to 2 ms and bandwidth above 100 Hz; 2 to 3% at P2 and P3 with 2 to 10 ms and 10 to 49 Hz). The host period matters beyond 20 ms, the bus delay hardly up to 10 ms. The PD re-tuned through each profile halves its stiffness and trades accuracy for compliance and smoothness; the velocity-feedback filter sweep bounds the raw-velocity assumption at a tenth of the torque rate, and removing the interface's sources one at a time through P1 orders them: the hold and the bus delay first (587 to 263 Nm/s in free motion), the feedback quantisation second (to 186 with both gone, against 115 at the torque interface), the command channels' 12 bits last and only once the others are gone.</p>"""

LEARNED_IFACE = """<p>A network per profile that takes the upper frame and the proprioceptive history at the host rate and emits bounded corrections on the channels the profile has (a goal offset, gain multipliers where writable, a feedforward residual), trained by backpropagation through the plant on the torque-tracking family against the realised torque at every physics step. On no profile does it halve the model-based interface's error (criterion i). The error-floor decomposition explains why: with the reference advanced to the middle of the hold, the model-based interface is at the floor the servo's own loop sets between frames (removing the hold and the delay together leaves 0.5% at P1, from the feedback and command quantisation); a scan of constant corrections finds nothing below it either. The lesson is a modelling one: the first version of the mapping did not advance the reference over the delay, and the learned interface's apparent gains in that version were the model's omission.</p>"""

STUDENTS = """<p>A fresh gain-residual network per profile (four layers of 256; a history of 100 ms at the host rate; the external-torque estimate; the feedforward PD base), distilled by DAgger through the profile from the ideal-robot labels in free space and the oracle in contact, deciding once per host frame with no compute latency and the bus delay kept. Through P1, P2 and P3 the students meet 25, 24 and 26 of 33 rows against the torque-interface classical set through the same profile, and 21, 22 and 18 once the PD re-tuned through the profile joins it: they win hold (1.3 to 1.5 against 8.2 mrad), steps, the task family and the pick (14 to 16 of 16 grasps against 7), keep free motion within 3 to 7%, and miss the blocked family and the fast free-space families through their torque rate. No training or evaluation episode diverged through any profile (an earlier reading to the contrary was an artefact of a flag, corrected in report 21's erratum).</p>"""

BLOCKED = """<p><strong>The blocked family at the host rate.</strong> The P1 student presses 23.5 N above the cap where the oracle through the same profile presses 12.7. Three diagnostics without training locate the miss. The label fit along the student's own episodes is good (a 2.4 Nm RMS error on a 16 Nm RMS yield in the blocked phase, correlation 0.98) and unbiased along the contact normal (a push error of 0.4 N mean, 7 N RMS), so the regression is not under-fitting the contact labels. The blocked phase in time (the table below) shows where the excess is: in the first half second after the impact (54% of it) and in the payload cases (7.0, 11.9 and 33.8 N at 0, 1 and 2 kg against the oracle's 7.9, 8.2 and 11.4; the student's steady press in the last half second is 16.3 N against the oracle's 14.0). The oracle knows the payload and compensates its weight along the normal; the student has to infer it, and the probes find the payload not linearly decodable from any layer. Weighting the contact rows three times made the family worse at every iteration (the selected checkpoint was the warm start), neutral weights were worse than the recipe, and the force-space label term dominated the regression and never validated below the PD. Keeping the teacher driving half the episodes for three more iterations did it: blocked 8.28 against the recipe's 11.74 (the oracle 7.41), every blocked row of every group 6 to 15% better than the best classical controller, hold and free motion kept, and 25 of 33 rows against the classical set with the re-tuned PD (the recipe 21). It falls short of the pre-registered excess target (19.7 against the oracle's 12.7 N above the cap) and loses the policy-like family and the fast free motion to the re-tuned PD through its torque rate. Over three seeds the effect holds in two (blocked 8.28, 10.14 and 11.27 against the recipe's 11.74) and the pooled variant meets 23 of 33 rows, so the blocked family at the host rate is improved on average and not settled; an eight-layer form of the same recipe never validated below the PD with this budget.</p>"""

YAM = """<p>The Menagerie YAM with its gripper baked into the tool body, torque motors at the DM motors' peak torques, the environment scaled to its reach, protocol v4 at its force scale. The shipped driver gains are a teleoperation setting (30 mrad on hold, 15 N above the cap); the CMA-ES re-tune through the interface halves the free-motion error and brings the press into the band. The host-rate student distilled from scratch through P1 (four layers of 256, one seed, selected at its third iteration) meets the criterion on 32 of 33 rows: hold 4 against 14 mrad, free motion at parity, blocked 5.1 against 9.5 N above the 12 N cap, 16 of 16 grasps against 9; the miss is blocked on the 500 N/m surface through its torque rate. Under the randomised DC-motor layer (torque-constant error ±15%, voltage envelope, friction, current-loop lag up to 2 ms, 12-bit torque feedback) it keeps every one of the 16 evaluated rows, and the layer's lag smooths every controller by about 40%.</p>"""

PROBES = """<p>Linear probes from the observation and from every trunk layer, replayed through the profile on validation episodes, with a nonlinear probe beside the linear one. The deeper host-rate trunks decode contact at 0.84 to 0.94 balanced accuracy (the two-layer torque-interface student 0.77 to 0.80), the reference's depth inside the surface at R² 0.75 to 0.90 (the quantity the impedance law needs), sliding against pressing at 0.77 to 0.82, and the blocked and release phases at 0.85 to 0.97; the surface normal in the tool frame is readable from the observation and, on the YAM, from the layers (R² 0.42 to 0.66, 0.80 nonlinearly); the stiffness classes, the payload and the contact force are not linearly decodable from any layer. The block-in-the-gap probe before the grasp reads 0.88 to 0.97, but so does the observation (0.90): it reads the plan's phase, not the block.</p>"""

CORRECTIONS = """<ul>
<li><strong>The model-based interface did not advance the reference over the bus delay and the hold.</strong> Found by the error-floor decomposition; fixed to the middle of the hold (zero-order-hold compensation); every Phase 1 and 2 table re-run; the pre-fix results kept under runs/m11/pre_fix and not reported.</li>
<li><strong>The torque-tracking metric's command ignored the reference's motion within a hold.</strong> Fixed: the command is the latest frame's law with the current reference at the true state, so the hold and the delay both count as error.</li>
<li><strong>The oracle's frame path indexed its reference by call count.</strong> Through a profile it ran four times too slow; fixed to time indexing and the oracle rows re-evaluated.</li>
<li><strong>Report 21's collection divergences were a flag's artefact.</strong> The free-space labelling path compared the labelled decisions with a tenth of the physics steps; at 50 Hz every episode tripped it. No episode diverged; the finding and its mechanism were withdrawn in an erratum, the flag now counts decisions.</li>
</ul>"""

LESSONS = """<ul>
<li><strong>Model the interface exactly before learning it.</strong> The learned interface's only apparent gain was the model's omission of the delay compensation; once the classical mapping was exact at the frame's arrival it sat at the servo's floor and the network found nothing below it. A learned interface is worth revisiting only on a servo whose loop the model gets wrong (friction, backlash, a firmware filter), which is what the DC-motor layer and hardware measurements are for.</li>
<li><strong>Rebuild the controller at the host rate rather than wrap the 1 kHz one.</strong> The host-rate student, distilled from scratch through the profile with its history at the host rate, recovered the torque-interface controller's wins on hold, steps, tasks and the pick through every profile; deeper trunks encode contact sharper than the two-layer student.</li>
<li><strong>The classical set must include the PD tuned through the profile.</strong> Softening is the classical answer to a position interface; it halves the blocked excess and the torque rate at the price of accuracy. The criterion's classical reference through a profile includes it.</li>
<li><strong>A one-sided metric around a target at the cap punishes variance.</strong> The student's push error along the normal is unbiased, yet the excess above the cap grows with its spread, most under payload at the impact; a label term in force space addresses what the torque loss cannot see.</li>
<li><strong>The torque rate through a position interface is structural.</strong> The held setpoint moves by the reference velocity times the period at every frame; neither a filtered velocity estimate nor a softer PD recovers the torque interface's smoothness. Only a firmware that interpolates the goal between frames would, and that is a measurement for later.</li>
<li><strong>Check flags against the rate they were written for.</strong> A divergence threshold in physics steps read every 50 Hz episode as diverged and produced a confident mechanism for a non-event.</li>
</ul>"""

NEXT = """<ul>
<li>The sustained-contact tasks proposed in the context document (wiping under a force band, drawer and door, chamfered peg insertion, an object along a wall, a lid, erasing with a held tool) are the task-level signal the milestone did not have.</li>
<li>Inference time: the simulation paused for the network this round; the host-rate students' four layers of 256 at 250 Hz are the budget to meet, by distillation into a smaller network or by measurement on the host.</li>
<li>The Dynamixel arm (Koch) on P2 with the XM430 constants identified on the public dataset, and with it the transfer check to a data-fitted actuator surrogate; deferred by the advisor's decision.</li>
<li>Hardware measurements when allowed: the bus period and delay, the velocity estimate the firmware damps on, and whether the goal is interpolated between frames.</li>
<li>A payload estimate as an explicit input (or a label term that makes the student infer it) for the blocked family under payload, where the oracle's privilege is the payload's weight along the normal.</li>
</ul>"""

if __name__ == "__main__":
    main()
