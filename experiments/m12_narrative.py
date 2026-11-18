"""Narrative of report 23 (Milestone 12, the consolidation on the torque-interface Panda; ADR 0012, protocol v5). The result
sections are filled from runs/m12/metrics_test.json (or LLC_METRICS)."""
import os as _os
from pathlib import Path

from experiments.d7_extras import LABELS
from experiments.m12_extras import (CLASSICAL_KEYS, STUDENTS, criterion_counts, criterion_table, deployment_table, depth_table, families_table, load_metrics, scale_table,  # noqa: F401
                                    task_table, with_reference)

_path = Path(_os.environ.get("LLC_METRICS", "runs/m12/metrics_test.json"))
_data, _groups = load_metrics(_path) if _path.exists() else ({}, {})
LEARNED = [k for k in STUDENTS if k in _groups.get("in_distribution", {})]
ORDER = ["best_classical", "pd_family", "pd_feedforward", "pd_shared_cap_feedforward", "oracle_pd"] + LEARNED
METHODS = [(k, LABELS.get(k, k)) for k in ORDER if k in _groups.get("in_distribution", {})]
_family_groups = with_reference(_groups, ("pd_family",)) if _groups and "pd_family" in _groups.get("in_distribution", {}) else {}
_counts_global = criterion_counts(_groups, LEARNED) if _groups else {}
_counts_family = criterion_counts(_family_groups, LEARNED) if _family_groups else {}

FAMILIES_SECTION = ("<h2>The fourteen families</h2>\n" + families_table(_groups, [k for k, _ in METHODS]) + task_table(_groups, ["best_classical", "pd_family", "oracle_pd"] + LEARNED)) if _groups else ""
CRITERION_SECTION = ("<h2>The criterion against the global classical set</h2>\n<p>ADR 0010's criterion through protocol v5: within 5% of the best global classical controller of each family (the two tuned PDs and their feedforward variants), per group; the gate of ADR 0012.</p>\n" + criterion_table(_groups, LEARNED)) if _groups and LEARNED else ""
FAMILY_SECTION = ("<h2>The criterion against the per-family PD</h2>\n<p>The strongest fixed-gain reference: a PD tuned on each family alone (CMA-ES on the cap composite, the feedforward form); the goal of ADR 0012, not its gate.</p>\n" + criterion_table(_family_groups, LEARNED, caption="The students against the per-family PD (the reference column names the family's own tuned PD).")) if _family_groups and LEARNED else ""
DEPLOY_SECTION = ("<h2>Deployment groups</h2>\n" + deployment_table(_groups, LEARNED)) if _groups and LEARNED else ""
DEPTH_SECTION = ("<h2>Depth</h2>\n" + depth_table(_data, _groups)) if _groups and "distill_consol_l10" in LEARNED and "distill_consol_l4" in LEARNED else ""
SCALE_SECTION = ("<h2>The data scale</h2>\n" + scale_table(_groups)) if _groups and "distill_consol_l4_small" in LEARNED else ""

NARRATIVE = {
    "question": """<p>Does the learned low-level controller of Checkpoints 2 and 3, trained on ten times the data under a plant distribution wide enough to cover a real Panda, hold its across-the-board result on a task set extended with contact-rich and sustained-contact work, and does depth buy anything there? The controller is the converged two-phase gain-and-residual student on the 1 kHz feedforward PD base; the comparison is with the global PD, the per-family PD and the privileged oracle, on protocol v5 at 32 cases per family (ADR 0012).</p>""",
    "setup": """<p><strong>Task set.</strong> The nine families of protocols v3 and v4 and five new ones: wiping (a raster under the force band on the table or the wall), erasing with a held block pressed on the table, peg-in-hole with a chamfer (a 20 mm square peg hanging from the fingertips, 0.75 or 1.5 mm of clearance per side, the reference 2 to 6 mm beside the hole), a drawer pushed closed to its stop along a reference that drifts sideways and continues past the stop, and a door pushed closed along a straight line while the contact slides on the panel. Thirty-two cases per family; the v3 and v4 held-out groups; and the deployment groups: the learned controllers' base at half and double stiffness, a 3 ms actuator lag with a 5 ms transport delay, compute latencies of 5 and 10 ms.</p>
<p><strong>Controllers.</strong> The global PD (tracking-tuned and cap-tuned, feedforward form), the per-family PD (CMA-ES per family on the cap composite, feedforward form), the oracle around the tracking-tuned feedforward PD (its plane per task: the table, the wall, the held block's contact plane, the hole's top plane with a weak lateral spring while the peg is blocked, the panels' stops), and the students: the free-space residual base trained through the plant (Milestone 10 round 2's recipe) at 4 layers of 256 and at 10 layers of 256 with residual blocks and layer normalisation, then DAgger from the ideal-robot labels in free space and the oracle in contact for twenty iterations of 320 episodes on the consolidation mixture of the fourteen families under the training distribution of protocol v5 (robot instance, payload unknown to the student, sensor noise, compute latency, actuation delay and lag, the base PD's stiffness and damping, surface stiffness and friction, object masses), with the base's gain scales and the latency as inputs; no cap on the label store; the teacher's share 1, 0.5, 0.25, then 0; validation on 96 cases over every family. The scale control is the 4-layer student at the Milestone 10 budget (12 iterations of 48 episodes), everything else equal. One seed each (the advisor's decision: seed 0 on the MacBook).</p>""",
    "extra_sections": [s for s in (FAMILIES_SECTION, CRITERION_SECTION, FAMILY_SECTION, DEPLOY_SECTION, DEPTH_SECTION, SCALE_SECTION) if s],
    "findings": """<ol>
<li>RESULTS_PENDING</li>
</ol>""",
    "caveats": """<ul>
<li>One seed per student; the pre-registered criteria of ADR 0012 ask for three, and the seed spread of Milestone 11 (blocked composites 8.3 to 11.3 across three seeds of one recipe) says one seed bounds nothing.</li>
<li>The per-family PD is tuned on four cases per family at the Milestone 9 budget; a larger budget would make it a harder reference.</li>
<li>The new families' metrics are the first of their kind here: the door's success threshold (2 degrees), the peg's (20 mm) and the drawer's (5 mm) were set before the runs and not tuned to any controller.</li>
</ul>""",
    "next": """<ul>
<li>Seeds 1 and 2 of the recipe that comes out, on the cluster or over following days.</li>
<li>The Cartesian impedance base and the sixth sustained-contact family (an object along a wall), deferred by the advisor.</li>
<li>Hardware (Milestone 7) remains unauthorised; the deployment bundle (experiments/m12_export.py) carries what a Panda needs.</li>
</ul>""",
}
