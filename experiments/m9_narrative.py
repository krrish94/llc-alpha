"""Narrative of report 14 (Milestone 9: plant v3 and protocol v3 baselines; ADR 0009). Result
sections are filled from runs/m9 after the evaluations."""
import json as _json
import os as _os
from pathlib import Path

from experiments.d7_extras import LABELS, load
from experiments.m9_extras import REPORT, fig_detachment, task_table

LABELS.update({"pd_shared": "PD, shared gains (re-tuned on plant v3)", "pd_shared_v2gains": "PD with the plant v2 gains", "oracle_pd": "Oracle around the PD (privileged)",
               "pd_feedforward": "PD + inverse-dynamics feedforward", "pd_residual_acmed_s02_lowrate": "PD + residual trained on plant v2 (D3)", "distill_gain_ext": "Distilled student trained on plant v2 (D6)"})
_metrics_path = Path(_os.environ.get("LLC_METRICS", "runs/m9/metrics_baselines.json"))
_data, _groups = load(_metrics_path) if _metrics_path.exists() else ({}, {})
ORDER = ["pd_shared", "pd_shared_v2gains", "pd_feedforward", "oracle_pd", "pd_residual_acmed_s02_lowrate", "distill_gain_ext"]
METHODS = [(k, LABELS.get(k, k)) for k in ORDER if k in _groups.get("in_distribution", {})]
_keys = [k for k, _ in METHODS]

DETACH_SECTION = "<h2>Light-contact detachment: sphere against pads</h2>\n<p>The plant finding of report 13 revisited on the gripper. Constant-force presses by the oracle law around the shared PD; the loss count is the protocol's (gaps of 5 ms or more).</p>\n" + fig_detachment({"v2": Path("runs/m9/detachment_v2.json"), "v3": Path("runs/m9/detachment_v3.json")})
TASK_SECTION = ("<h2>Task families</h2>\n<p>Box pushing and pick-and-place with a held block (16 test cases each); the shared PD and the privileged oracle set the first marks.</p>\n"
                + task_table(_groups, _keys, "Task families on plant v3: success rates and the task metrics (means over 16 cases).")) if "tasks" in _groups else ""
_transfer_path = Path("runs/m9/metrics_transfer.json")
if _transfer_path.exists():
    from experiments.d7_extras import STOP_WAIT_KEYS_V3, stop_and_wait_table

    _tdata, _tgroups = load(_transfer_path)
    _tkeys = [k for k in ("pd_shared", "pd_shared_v2gains", "oracle_pd", "pd_residual_acmed_s02_lowrate", "distill_gain_ext") if k in _tgroups.get("in_distribution", {})]
    TRANSFER_SECTION = ("<h2>Transfer from the sphere plant: the D3 residual and the D6 student on the gripper</h2>\n"
                        "<p>The controllers trained on plant v2 (with the v2 gains as their base, which they carry) run unchanged on plant v3 under protocol v3: in-distribution families and the task families, 16 cases each, seeds of the student pooled.</p>\n"
                        + stop_and_wait_table(_tgroups, _tkeys, "Blocked targets on plant v3, controllers trained on plant v2 against the v3 baselines (apparent stiffness per metre of reference depth, contact stiffness per metre of actual error; release reaction and settling are the protocol v3 metrics).", keys=STOP_WAIT_KEYS_V3)
                        + task_table(_tgroups, _tkeys, "Task families on plant v3 for the same controllers."))
else:
    TRANSFER_SECTION = ""
_videos = REPORT / "data" / "videos.html"
VIDEO_SECTION = ("<h2>Representative episodes</h2>\n" + _videos.read_text()) if _videos.exists() else ""

NARRATIVE = {
    "question": """<p>Milestone 9 moves the work to the robot the task-level evaluation should target: the DROID robot, a Franka Panda with a Robotiq 2F-85. This report establishes the new plant and protocol v3, and sets the baselines against which the retrained controllers will be read. Three questions: does the sphere plant's light-contact detachment persist on fingertip pads; what do the re-tuned PD, the feedforward PD and the privileged oracle score on the v3 families, the compliant surfaces and the two task families; and do the controllers trained on the sphere plant transfer to the gripper.</p>""",
    "setup": """<p><strong>Plant v3 (ADR 0009).</strong> The Menagerie 2F-85 is settled at a commanded opening in a standalone simulation and baked, with every geom at its settled pose and the merged inertia (1.053 kg at the flange, closed), into one rigid tool body on the Panda flange; the controlled system stays at seven DoF. The tool point is the centre of the two fingertip end faces, the four pad boxes are the tool geoms with the contact model of the sphere tool (the Menagerie pad parameters gave 400 to 1600 N impacts), every other gripper geom is a robot link. The gripper is not controlled: closed for the surface families and for pushing, open to 38 mm for pick-and-place.</p>
<p><strong>Protocol v3.</strong> The v2 families, splits and held-out groups with: press depths of 0.4 to 1.4 cm (7 to 12 N under 5 N + 500 N/m x depth) and a press floor of 7 N; the blocked composite with the release reaction (time after release until the tool moves toward the reference faster than 2 cm/s) and the release settling (time after the reference reaches its final waypoint until the error is below 1 cm) in place of the recovery; link clearance for every surface family by signed distance with the lift correction (2 mm of real clearance); table tilt 0 to 10 degrees and wall tilt 0 to 15 degrees. The shared PD gains are re-tuned on v3 by the Milestone 1 procedure. Two task families: box pushing (a 5 cm, 0.2 kg box pushed 10 cm with the fingertip faces, the tool axis 25 or 40 degrees below horizontal) and pick-and-place (a 34 mm, 0.3 kg block held by a weld, carried, placed 2 mm into the table, released 0.2 s into the dwell, lifted). Nothing from protocol v2 is compared with these numbers.</p>""",
    "extra_sections": [s for s in (DETACH_SECTION, TASK_SECTION, TRANSFER_SECTION, VIDEO_SECTION) if s],
    "findings": """<ol>
<li><strong>The re-tuned PD is a stiffer controller than the sphere plant's.</strong> The Milestone 1 procedure on plant v3 (CMA-ES on the v2 composite over four tune cases per family) returns gains two to nine times those of the sphere plant (joint 4: 1276 against 141 Nm/rad). In free space it tracks at 8.8 mrad against 19.8 for the v2 gains on the same plant, holds against 10 N pulses with 8 mm of deflection against 16 mm, and pays with a torque-rate RMS of 115 against 53 Nm/s. In contact it pushes harder: 41 N above the cap on blocked targets against 30 N, apparent stiffness 1440 against 1260 N/m, presses at 22 N where the band asks for 7 to 12 N. The feedforward PD tracks free motion at 4.9 mrad, the best classical number of the protocol.</li>
<li><strong>The release metrics do what they were defined for.</strong> The oracle around the re-tuned PD stops and waits (2.6 N above the cap, no drift, moving 1% of the time, apparent stiffness 418 N/m) and now scores the same release reaction (0.05 s) and settling (0.28 s) as the PD, where the old recovery metric charged it 1.18 s against 0 for leaving the tool behind the reference. Its blocked composite is 4.7 against 13.8.</li>
<li><strong>Light contact detaches on the pads too; the sweep makes it a property of force, speed and payload, not of the point contact.</strong> Constant-force presses of the oracle law on four table cases (losses of 5 ms or more per second of slide): at the protocol's nominal speed without payload the sphere tool loses contact 21 times per second at 5 N, 6 at 15 N and none at 30 N; the gripper pads 20, 15 and 11. At half speed both reach zero or near it by 15 to 20 N; at twice the speed both keep 4 to 20 losses per second up to 30 N, and a 2 kg payload at twice the speed keeps 10 to 20 on both. So fingertip pads do not remove the detachment, the 7 to 15 N band of the cap sits inside the detaching regime at the nominal slide speed on either plant, and the loss metric of the two sliding families measures the simulator's sliding dynamics as much as the controller. On the press-and-slide family the oracle presses at 9.7 N and loses contact 20 times per second against 6.7 for the PD at 22 N.</li>
<li><strong>The sphere-plant controllers transfer to the gripper.</strong> Run unchanged on plant v3, with the v2 gains they carry as their base, the D6 student stops and waits (2.3 N above the cap against 40.7 for the re-tuned PD and 30.4 for its own base, drift 1 mm, moving 9% of the time, apparent stiffness 329 N/m, release reaction 0.10 s and settling 0.23 s against the PD's 0.05 and 0.28), presses in band on the contact family (8.6 N, 2.6 N above the cap) and at the cap on the task family, and its in-distribution mean composite is 2.82 against 3.06 for the privileged oracle around the re-tuned PD, 3.52 for the D3 residual and 4.55 for the PD. Its free space is that of its base, not of the stiffer v3 PD (hold 2.46 against 1.74, free motion 1.01 against 0.56), the one place the re-tuned PD wins. The gripper changes the tool's mass (1.05 against 0.7 kg) and its contact geometry; the behaviour learned on proprioception carried over without a single rollout on the new plant. Distilling the transferred students further on the gripper (six DAgger iterations with the oracle around themselves as the teacher) adds nothing: two of three seeds keep the unchanged student as their best checkpoint on the 32 validation cases, the third gains 0.05.</li>
<li><strong>The task families separate controllers.</strong> Pushing a box 10 cm with the fingertip faces: the stiff PD succeeds in 31% of the cases, the soft v2 gains in 56%; both push the right distance (errors 0.3 to 0.6 cm) and fail on the box's rotation (mean yaw 20 and 15 degrees against the 15-degree criterion), so pushing is a question of how the contact is held, not of tracking. Pick-and-place: the stiff PD places every block within 2 mm with a 14 N peak on the table, the soft gains place half of them (2.2 cm mean error, 23 N peak) and track the lift after the release twice as badly (16 against 9 mrad). The oracle is the PD on these families by construction (no surface is pressed). The transferred student pushes with 48% success (its base: 56%; the stiff PD: 31%) and places every block within 1 cm with a 6 N peak, less than half the stiff PD's 14 N, with the lift tracked at 11 mrad: the learned controller is the gentlest placer and keeps its base's pushing.</li>
</ol>""",
    "caveats": """<ul>
<li>Baselines only. Nothing here is compared with the sphere plant's tables; the gains, the plant and three definitions changed at once.</li>
<li>The PD tuning objective is still the protocol v2 composite (as in Milestone 1 and D1); it rewards stiffness and does not see the cap. A tuning on the cap composite would give a softer PD; it was not done, to keep the procedure constant across plants.</li>
<li>The task families' force metrics are taken on the object (push) and on the block-table contact (pick); the gripper is not controlled, so pick-and-place measures the carry, the placement and the payload change only.</li>
</ul>""",
    "next": """<ul>
<li>The residual warm start on plant v3 by the D3 recipe (running), the distilled students with the free-space fix (three seeds), the TAM-style baseline; report 15 with the retrained controllers against these baselines.</li>
<li>Decide whether the sliding families' loss metric stays as is, given the detachment sweep.</li>
</ul>""",
}
