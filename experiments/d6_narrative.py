"""Narrative of deep-dive report D5 (contact maintenance under a 15 N cap); result sections are
filled after the stage evaluations."""
from pathlib import Path

from experiments.d6_extras import LABELS, audit_table, method_key, training_curves

import json as _json
import os as _os

METHODS = [("pd_shared", "PD, shared gains (v2)"), ("pd_residual_acmed_s02_lowrate", "PD + residual, actor-critic (D3 warm start)")]
_metrics_path = Path(_os.environ.get("LLC_METRICS", "runs/d6/metrics_test.json"))  # the report's tables cover the arms evaluated into this file
_evaluated = None
if _metrics_path.exists():
    _learned = _json.loads(_metrics_path.read_text())["learned"]
    _evaluated = {info.get("method_key") or info["policy_kind"] + (f"_{info['variant']}" if info.get("variant") else "") for info in _learned.values()}
_runs = []
for run in sorted(Path("runs/d6").glob("*-s*")) if Path("runs/d6").exists() else []:
    if run.is_dir() and (run / "meta.json").exists() and not run.name.startswith("killed"):
        key = method_key(run)
        if key not in dict(METHODS) and (_evaluated is None or key in _evaluated):
            METHODS.append((key, LABELS.get(key, key)))
        _runs.append(str(run))
AUDIT_SECTION = "<h2>Reward audit</h2>\n<p>Before training, the training reward is evaluated on identical windows for fixed controllers, including a privileged exemplar of the wanted behaviour (the shared PD with its gains scaled to 0.3 only while the reference is inside the surface). A reward that does not rank the exemplar first cannot be optimised toward it.</p>\n" + audit_table()
TRAINING_SECTION = ("<h2>Training</h2>\n<p>One line per run; seeds share a colour.</p>\n" + training_curves(_runs)) if _runs else ""
_videos = Path("research/reports/11-dd-contact/data/videos.html")
VIDEO_SECTION = ("<h2>Representative episodes</h2>\n<p>One case per family (median composite of the protocol v2 PD); contact families also at quarter speed.</p>\n" + _videos.read_text()) if _videos.exists() else ""

NARRATIVE = {
    "question": """<p>Every learned controller of reports D3 and D4 tracks better and limits impact forces better than the tuned PD, and every one of them is worse at staying in contact: contact losses during a press, the press deficit, and the recovery after a blocked reference is released. The saved episodes show a learned flinch at the reward's soft force limit, and the training reward evaluated on saved episodes prefers yielding on blocked targets (which no optimiser found) and intermittent light contact on the task family (which the reward asked for). The advisor's directive: train the residual and PPO arms in parallel, audit the reward on the situations that matter, set the force cap to what pick-and-place, peg insertion and connector tasks need, and run a contact-focused study that also examines the model: what the policy needs to sense contact, what output parameterisation produces compliance, whether the feature encoding matters, and whether the outputs are smooth enough for hardware.</p>
<p>Three defects of the harness were found on the way and are corrected here (ADR 0007): table-surface blocked references had no blocked phase at all (their training windows contained no block, their recovery metric was undefined); in the wall branch of the blocked and contact families the hand and the forearm reached the wall before the tool, so those cases measured unsensed link collisions; and two validation cases per family had misranked the D4 contact arms. The wall cases are now a genuine tool block (tool contact 93 to 98% of the blocked phase, no link contact), and they are hard: the tuned PD pushes with 50 to 200 N against the wall.</p>""",
    "setup": """<p><strong>Force cap.</strong> Sustained contact force capped at 15 N, a hardware-like collision reaction modelled at 25 N of the 20 ms filtered force. Connector datasheets list 20 to 30 N mating forces, third-party Franka software defaults its collision force threshold to 30 N; pick-and-place and compliant peg insertion need a few newtons. New protocol metrics: excess force above 15 N, contact losses per second of press phase, and a composite under the cap (the v2 composite with the force terms at the cap and the sustained peak); the v2 metrics stay for continuity.</p>
<p><strong>Reward.</strong> The D3 actor-critic reward with the force penalty from 15 N, plus privileged contact-maintenance terms while the reference is inside a surface: a 5 N Gaussian kernel on the normal force around the force the commanded depth implies (5 N + 1000 N/m x depth, capped at 15 N), the tangential tool-tracking kernel (1 cm), and the joint kernel switched off (it asks for an infeasible depth). A lost contact scores about zero on the force kernel. The actor stays proprioceptive; the terms use the reference's tool path, the surface normal and the contact force, all simulation quantities.</p>
<p><strong>Parameterisations.</strong> The torque residual of D3 (tanh-bounded, half the torque limit) against a gain residual: bounded per-joint multipliers of the shared PD gains (stiffness 0.02 to 1.5, damping 0.3 to 2.0, neutral at 1) plus the same torque residual. A D3 checkpoint converts exactly to the gain residual (gain heads neutral), so every arm starts from the same controller.</p>
<p><strong>Windows and mixture.</strong> Contact-focused mixture (hold 0.1, free 0.15, contact 0.35, blocked 0.4; the task family stays held out). Half of the surface windows start 0.1 to 0.2 s before the reference crosses the surface, so the window is spent on the block and, with 4 s windows, on the release. Both training methods warm-start from the D3 residual (runs/d4/pd_residual_acmed_s02_lowrate-s0). Actor-critic: 450 iterations of 24 windows of 2 s, finite-difference plant gradients with median clipping, as in D3. PPO: 200 updates of 64 windows of 4 s, 10 ms decisions, AR(1) exploration noise (correlation 0.5), the 25 N filtered reflex charged per crossing without ending the window, no drift penalty. Validation every 50 iterations or 25 updates on 32 fixed cases (two per free-space family, eight per surface family), selection on the composite under the cap.</p>
<p><strong>Stage 1 arms.</strong> {actor-critic, PPO} x {cap reward, D3 reward} x {torque residual, gain residual}: three seeds for the cap arms, one for the D3-reward controls. Stage 2 adds, on the better parameterisation, an external-torque estimate from the robot's own model as an input (tau applied minus M qdd minus bias, the signal the Panda exposes on hardware), random Fourier features, and a 20 ms first-order filter on the held output.</p>""",
    "extra_sections": [AUDIT_SECTION] + ([TRAINING_SECTION] if TRAINING_SECTION else []) + ([VIDEO_SECTION] if VIDEO_SECTION else []),
    "findings": """<ol>
<li><strong>The cap reward names the behaviour; the D3 windows did not contain it.</strong> On identical training windows of the surface families, the cap reward ranks the privileged exemplar (stiff in free space, gains x0.3 only while the reference is inside the surface) first on both families and the D3 residual last, by a wide margin on blocked targets (-0.09 against -1.52 reward units per step). The D3 reward on the same windows also ranks the exemplar first, which it did not on D3's own windows: there, 60% of the blocked windows contained no block (erratum) and the rest were mostly approach. The failure of D3 and D4 on blocked targets was therefore first a data problem and then a reward problem, and only after both an optimisation problem.</li>
<li><strong>Stage 1, PPO arms (test tables, selected checkpoints, three seeds pooled for the cap arms).</strong> The D3-reward controls never beat their warm start on validation, so their selected checkpoint is the warm start itself. The cap-reward arms do: composite under the cap 4.79 (gain residual) and 4.83 (torque residual) against 4.96 for the warm start and 5.95 for the PD, from lower forces on the blocked target (excess above 15 N 51 N against 57 N; the PD 46 N) and on the contact family (11 to 12 N against 12.5 N; the PD 16 N), at a small cost in free motion (torque-rate RMS 68 to 72 against 44.5 Nm/s). The press deficit grows again (contact family 0.6 to 0.7 against 0.33; task 1.9 to 2.1 against 1.45) and contact losses do not improve. No arm is compliant: the apparent stiffness on blocked targets is 1300 N/m against 1440 for the warm start and 1180 for the PD, and the gain-residual arm's multipliers never left neutral (probe), so PPO at exploration std 0.05 did not explore the gain head at all.</li>
<li>Pending: stage 1 actor-critic arms and stage 2 (external-torque input, wider exploration).</li>
</ol>""",
    "caveats": """<ul>
<li>The cap puts the protocol's task family partly out of scope by design: a press commanded 1 cm deep asks for 15 N, and the family's 30 N-class presses are no longer a target.</li>
<li>The contact-maintenance terms, the critic and the exemplar of the audit use simulation-only information; the actor sees proprioception and the position target only.</li>
<li>One seed for the control arms; three for the cap arms. The test tables (16 cases per family) are the evidence; validation curves are a screen.</li>
</ul>""",
    "next": """<ul>
<li>Stage 2 on the better parameterisation: external-torque input, Fourier features, output filter.</li>
</ul>""",
}
