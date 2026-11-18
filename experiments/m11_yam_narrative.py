"""Narrative of report 20 (Milestone 11, Phase 3: the YAM through its native interface; ADR 0011 decision 8). The result sections
are filled from the metrics file named by LLC_METRICS (default runs/m11/yam/metrics_p1_student.json: the classical set, the
oracle ceiling and the host-rate student through profile P1 on plant ``yam``) and from the tune record runs/m11/yam/tune_p1."""
import json
import os as _os
from pathlib import Path

from experiments.d7_extras import LABELS, load
from experiments.d_report import add_best_classical, wins_table
from experiments.m10_extras import criterion_table, torque_rate_table
from experiments.m11_extras import torque_table

LABELS.update({"pd_shared": "PD re-tuned on the YAM (CMA-ES through P1, cap composite) + inverse-dynamics feedforward", "pd_shipped": "PD with the i2rt driver's shipped gains", "pd_feedforward": "Re-tuned PD + inverse-dynamics feedforward (same as the re-tuned set)",
               "pd_shipped_feedforward": "Shipped gains + inverse-dynamics feedforward", "best_classical": "Best classical controller of the family", "oracle_pd": "Oracle around the re-tuned PD (privileged ceiling)",
               "distill_yam_p1": "Host-rate two-teacher student (YAM, P1, 4 layers of 256)", "torque_program": "Torque program on the re-tuned feedforward PD"})
_metrics_path = Path(_os.environ.get("LLC_METRICS", "runs/m11/yam/metrics_p1_student.json"))
_data, _groups = load(_metrics_path) if _metrics_path.exists() else ({}, {})
for _g in _groups.values():
    _g.pop("pd_feedforward", None)  # the re-tune ran in the feedforward form, so the re-tuned set already carries the feedforward: one row
    add_best_classical(_g)  # the shipped gains, with and without feedforward, and the re-tuned set are the classical candidates
ORDER = ["best_classical", "pd_shipped", "pd_shipped_feedforward", "pd_shared", "oracle_pd", "distill_yam_p1"]
METHODS = [(k, LABELS.get(k, k)) for k in ORDER if k in _groups.get("in_distribution", {})]
LEARNED = [k for k in ("distill_yam_p1",) if k in _groups.get("in_distribution", {})]


def _gains_table() -> str:
    import sys

    sys.path.insert(0, str(Path("research").resolve()))
    from report_api import table

    rows = []
    shipped = Path("runs/m11/yam/gains_shipped.json"); tuned = Path("runs/m11/yam/tune_p1/gains_shared.json")
    if shipped.exists():
        s = json.load(open(shipped))["spec"]; rows.append(["Shipped (i2rt yam_v1)"] + [f"{a:.0f} / {b:.1f}" for a, b in zip(s["kp"], s["kd"])] + ["—"])
    if tuned.exists():
        t = json.load(open(tuned)); s = t["spec"]; rows.append([f"Re-tuned (CMA-ES, {t['iters']} iterations, {t['episodes_evaluated']} episodes)"] + [f"{a:.0f} / {b:.1f}" for a, b in zip(s["kp"], s["kd"])] + [f"{t['objective']:.2f} (initial {t['x0_objective']:.2f})"])
    return table(["Gains (kp Nm/rad / kd Nm s/rad)"] + [f"joint {j + 1}" for j in range(6)] + ["Tune objective"], rows, caption="The YAM's PD gains: the i2rt driver's shipped values and the CMA-ES re-tune through profile P1 on the cap composite of the tune split (four cases per family, the Milestone 9 budget). The shipped set is the manufacturer's teleoperation setting, not a tracking tune.") if rows else ""


GAINS_SECTION = "<h2>The classical gains</h2>\n" + _gains_table()
CRITERION_SECTION = ("<h2>The success criterion</h2>\n<p>Every family and group composite of the student against the best classical controller of that family through the same interface (ADR 0011, criterion iv).</p>\n" + criterion_table(_groups, LEARNED)) if _groups and LEARNED else ""
WINS_SECTION = ("<h2>Where the student wins and loses</h2>\n" + wins_table(_data, METHODS, "best_classical")) if _data and LEARNED else ""
TORQUE_SECTION = ("<h2>The torque-tracking family</h2>\n" + torque_table({"p1": _groups}, caption="Torque-tracking family on the YAM through P1 (the re-tuned feedforward PD plus the torque programs, forces and torques scaled to the DM motors' peak torques).")) if "torque_tracking" in _groups else ""
SMOOTH_SECTION = ("<h2>Smoothness</h2>\n" + torque_rate_table(_groups, [k for k, _ in METHODS])) if _groups else ""
_videos = Path("research/reports/20-m11-yam/data/videos.html")
VIDEO_SECTION = ("<h2>Representative episodes</h2>\n" + _videos.read_text()) if _videos.exists() else ""
from experiments.m11_extras import motor_table, probe_table, transient_table  # noqa: E402

_motor = motor_table(Path("runs/m11/yam/metrics_p1_student.json"), Path("runs/m11/yam/metrics_p1_motor.json"), ["pd_feedforward", "pd_shipped_feedforward", "oracle_pd", "distill_yam_p1"])
_probe_yam = Path("runs/m11/yam/probe_ext_student_p1.json")
PROBE_SECTION = ("<h2>What the student's representations encode</h2>\n<p>Linear probes from the observation and from every trunk layer of the YAM student, replayed through P1 on validation episodes (report 19's study), with the extended targets of ADR 0011 amendment 3: a nonlinear probe beside the linear one, the reference's depth inside the surface, sliding against pressing, the phases of a blocked episode, and the block in the grasp tolerance before the grasp.</p>\n"
                 + probe_table(_probe_yam, caption="Probes of the YAM student through P1, layer by layer (four layers of 256).") + probe_table(_probe_yam, extended=True)) if _probe_yam.exists() else ""
_transient = transient_table(Path("runs/m11/yam/blocked_transient_p1.json"), cap=12.0)
TRANSIENT_SECTION = ("<h2>The blocked family in time</h2>\n<p>The normal force over the blocked phase of the sixteen in-distribution cases, split in time, with the excess above the 12 N cap by payload (the same split as the Panda's in report 21).</p>\n" + _transient) if _transient else ""
MOTOR_SECTION = ("<h2>The DC-motor layer</h2>\n<p>Decision 9 in its robustness form (ADR 0011 amendment 3): under the servo law sits a DC-motor layer whose unknowns are drawn per case, since the YAM's motors have no identification data: the firmware's torque constant against the true one (±15%), the voltage envelope (the stall torque 1.5 to 3 times the peak, the no-load speed 0.8 to 1.2 times the velocity limit; braking torque unlimited), viscous friction up to 5% of the peak torque at the velocity limit, Coulomb friction up to 2% of the peak beyond the plant's own, a current-loop lag up to 2 ms, and a 12-bit torque feedback. The upper layers see the firmware's torque feedback, the commanded current times its own constant. The same controllers, the same cases.</p>\n" + _motor) if _motor else ""

NARRATIVE = {
    "question": """<p>The first robot of the milestone is the i2rt YAM, a 6-axis arm on Damiao motors whose host interface is an impedance frame on CAN at 250 Hz (profile P1). Everything on this page is the YAM through that interface with the model-based mapping of ADR 0011: the question is whether the protocol, the classical set and the learned controller carry over to a position-interface arm with a quarter of the Panda's torque and a 12 N force cap, and whether a host-rate student distilled from scratch through the interface meets the across-the-board criterion against the best classical controller of each family.</p>""",
    "setup": """<p><strong>Plant.</strong> The Menagerie YAM (MIT licence) with its gripper baked at a fixed opening into one rigid tool body (closed for the surface families and the push, 46 mm for the pick), torque motors on the six joints capped at the motors' peak torques (28 Nm on the first three, 10 Nm on the wrist), the Menagerie armature and 0.1 Nm of dry friction per joint, the model's own gravity compensation switched off (the controllers compensate gravity), and the protocol's environment scaled to the 0.78 m reach: a 6 cm table block and a wall at 0.58 m. The wrist points straight down only close to the base, so the table region is nearer than the Panda's and the blocked and task families tilt the tool 10 to 25 degrees forward.</p>
<p><strong>Protocol v4 scaled to the YAM</strong> (ADR 0011, decision 7d): force cap 12 N, press band 5 to 8 N (the press law 5 N + 500 N/m times a commanded depth of 0 to 6 mm), 5 N hold pulses, evaluation payloads of 0, 0.5 and 1 kg at the tool, the heavy-payload group at 1.5 kg, instance deviations as on the Panda. The families, splits, held-out groups and composites are those of protocol v3.</p>
<p><strong>Classical set.</strong> The i2rt driver's shipped gains (kp 80/80/80/10/10/10, kd 5/5/5/1.5/1.5/1.5) with and without the inverse-dynamics feedforward of the reference, and the PD re-tuned by CMA-ES through P1 on the cap composite in the feedforward form (40 iterations of 12 candidates on four tune cases per family, the Milestone 9 budget); the best of the three per family is the reference; the oracle around the re-tuned PD is the ceiling.</p>
<p><strong>Student.</strong> A fresh gain-residual network (four layers of 256, a history of 25 host frames at 4 ms, the external-torque estimate from the robot model, the feedforward base with the re-tuned gains) distilled by DAgger from two teachers through the interface: the ideal-robot labels in free space and the oracle in contact with the YAM's 5 N floor, twelve iterations of 48 episodes, the teacher driving in the first two, free-space labels weighted three times. It decides at every host frame with no compute latency; the bus delay stays.</p>""",
    "extra_sections": [s for s in (GAINS_SECTION, CRITERION_SECTION, WINS_SECTION, TORQUE_SECTION, SMOOTH_SECTION, TRANSIENT_SECTION, MOTOR_SECTION, PROBE_SECTION, VIDEO_SECTION) if s],
    "findings": """<ol>
<li><strong>The protocol and the classical set carry over, and the YAM's gains needed re-tuning.</strong> Every family, group and task of protocol v4 runs on the YAM through its 250 Hz impedance interface with no divergence. The i2rt driver's shipped gains are a teleoperation setting: 30 mrad on hold, 75 mrad in free motion, 50 mrad on blocked targets, 15 N above the 12 N cap, one grasp in five. The CMA-ES re-tune through the interface (kp 148/61/121/143/52/23 Nm/rad against the shipped 80/80/80/10/10/10) halves the free-motion error (29 mrad), brings hold to 14 mrad, the press into the 5 to 8 N band (8.3 N mean), blocked targets to 9.5 N above the cap, and the grasp to 9 of 16; the feedforward adds nothing to the shipped gains on the families the YAM's small gravity torques dominate. The torque-tracking family through the YAM's own interface puts the realised torque within 0.1 to 1.3% of the motors' peak torque with a 1 to 2 ms delay and a chirp bandwidth of 94 to 125 Hz, as on the Panda's P1.</li>
<li><strong>The host-rate student meets the across-the-board criterion on 32 of 33 rows, in one seed.</strong> Distilled from scratch through the interface in twelve iterations (its best validation at the third, 1.78 against 2.41 for its warm start), the student is within 5% of or better than the best classical controller on every family and group but one: hold 1.01 against 2.69 (4 against 14 mrad), free motion at parity (28 against 29 mrad, 1.79 against 1.77), policy-like targets at parity, steps 2.06 against 3.06, press-and-slide 1.08 against 1.29 (pressing at 6.7 N inside the band), blocked targets 3.38 against 5.19 (5.1 against 9.5 N above the cap, contact stiffness 449 against 581 N/m) and 3.61 for the oracle, the task family 1.00 against 1.92 and 1.70 for the oracle. It holds across the held-out groups (free motion on the held-out workspace 1.70 against 2.09, the 1.5 kg payload 3.73 against 4.16 with blocked 5.13 against 8.36, the doubled instance deviations 3.21 against 5.17 on blocked targets, the compliant surfaces 0.69 to 2.63) and on the tasks: 16 of 16 grasps against 9 (placing every block, the family composite 0.71 against 4.18) and 6 of 16 pushes against 4. The one miss is the blocked family on the 500 N/m surface, +22% against the shipped gains, which win that row with a torque rate of 43 Nm/s against the student's 173 while the student presses less (0.2 against 1.3 N above the cap).</li>
<li><strong>Where the student pays.</strong> Its torque rate is above the best classical controller's on every family (free motion 392 against 325 Nm/s, policy-like targets 1293 against 998, blocked 148 against 116), the price of the gain multipliers and the residual on a coarse interface; and it beats the oracle only where the oracle's own frame path through the interface softens it (blocked 3.38 against 3.61, the task family 1.00 against 1.70), not on the compliant surfaces (2.39 and 2.63 against 1.46 and 1.45 on blocked targets).</li>
<li><strong>Under a randomised DC-motor layer the student keeps every row.</strong> With the actuator's unknowns drawn per case under the servo law (the ranges of the section above), the student stays within 5% of or better than the best classical controller on all 16 rows of the evaluated groups (in distribution, the 500 N/m surface, the tasks, the 1.5 kg payload), no case diverges, and the classical set moves as little as the student does: the layer's lag smooths every controller (free-motion torque rate 325 to 170 Nm/s for the feedforward PD, 392 to 208 for the student), its friction and gain error cost 3 mrad of free-motion tracking (29.0 to 32.3 and 27.9 to 31.5 mrad), the blocked excess moves from 9.5 to 9.3 and from 5.1 to 4.8 N, the grasps stay at 9 and 16 of 16. A controller distilled on the servo abstraction is not fragile to the actuator physics the abstraction omits, within these ranges. The transfer check proper, to a data-fitted actuator surrogate, waits for the Dynamixel arm whose servo the public data identifies.</li>
</ol>""",
    "caveats": """<ul>
<li>The YAM plant is datasheet-grounded (motor torque limits, the Menagerie inertias and armature) with no identification data; the DC-motor layer is a randomised model whose ranges are guesses around the datasheets, not a fit, so the robustness result bounds the sensitivity to those effects and does not predict the arm.</li>
<li>One seed of the student; the classical gains tuned once; sixteen cases per family.</li>
<li>The environment regions and tilts were chosen for reachability, not from any task of the YAM's users.</li>
</ul>""",
    "next": """<ul>
<li>The selection rule's early optimum (the third iteration) is worth examining: the later iterations lose blocked targets on validation (6.3 to 10.8 against 4.6), the drift of the student-driven iterations seen on the Panda (report 21). Further seeds of the YAM student and the Koch arm are deferred by the advisor's decision of 2026-11-17.</li>
<li>The push family's success (6 of 16) is the task policy's line, as on the Panda; the sustained-contact tasks proposed in the context document are the next task-level signal.</li>
</ul>""",
}
