"""Narrative of report 21 (Milestone 11, Phase 2b: host-rate students through the profiles on the Panda; ADR 0011 decision 5).
The result sections are filled from runs/m11/metrics_students_<profile>.json (the classical set, the oracle and the student
through one profile) and runs/m11/probe_student_<profile>.json; the standard group sections show the profile named by
LLC_PROFILE (default p1)."""
import json
import os as _os
from pathlib import Path

from experiments.d7_extras import LABELS, load
from experiments.d_report import add_best_classical, wins_table
from experiments.m10_extras import criterion_table, torque_rate_table
from experiments.m11_extras import PROFILE_LABELS, diag_table, probe_table, transient_table, variants_table

_profile = _os.environ.get("LLC_PROFILE", "p1")
LABELS.update({"pd_shared": "PD tuned on the tracking composite", "pd_shared_cap": "PD tuned on the cap composite", "pd_feedforward": "PD (tracking-tuned) + inverse-dynamics feedforward",
               "pd_shared_cap_feedforward": "PD (cap-tuned) + inverse-dynamics feedforward", "best_classical": "Best classical controller of the family", "oracle_pd": "Oracle around the PD (privileged ceiling)",
               "pd_retuned": "PD re-tuned through the profile + inverse-dynamics feedforward",
               **{f"distill_host_{p}": f"Host-rate two-teacher student through {PROFILE_LABELS[p]}" for p in ("p1", "p2", "p3")},
               "distill_host_p1_w1c3": "P1 student, label weights free 1 / contact 3", "distill_host_p1_w1c1": "P1 student, neutral label weights", "distill_host_p1_beta": "P1 student, teacher share 0.5 for three more iterations",
               "distill_host_p1_f1": "P1 student, force-space label term (weight 1)", "distill_host_p1_f03": "P1 student, force-space label term (weight 0.3)"})
# through P1 the Phase 2c and 2d files carry the recipe's student, its variants and the re-tuned PD; the earlier file otherwise
_metrics_path = next((Path(f) for f in (f"runs/m11/metrics_phase2d_{_profile}.json", f"runs/m11/metrics_phase2c_{_profile}.json", f"runs/m11/metrics_students_{_profile}.json") if Path(f).exists()), Path(f"runs/m11/metrics_students_{_profile}.json"))
_data, _groups = load(_metrics_path) if _metrics_path.exists() else ({}, {})
for _g in _groups.values():
    add_best_classical(_g)
STUDENT = f"distill_host_{_profile}"
LEARNED = [k for k in ([STUDENT] + sorted(_groups.get("in_distribution", {}))) if k.startswith("distill_host_") and k in _groups.get("in_distribution", {}) and k not in ([STUDENT] if k != STUDENT else [])]
LEARNED = list(dict.fromkeys(LEARNED))
ORDER = ["best_classical", "pd_shared", "pd_shared_cap", "pd_feedforward", "pd_shared_cap_feedforward", "pd_retuned", "oracle_pd"] + LEARNED
METHODS = [(k, LABELS.get(k, k)) for k in ORDER if k in _groups.get("in_distribution", {})]


def _across_profiles() -> str:
    """The student against the best classical controller and the oracle on every family, one block per profile."""
    import sys

    sys.path.insert(0, str(Path("research").resolve()))
    from report_api import table

    from experiments.d7_extras import mean_metric

    rows = []
    for p in ("p1", "p2", "p3"):
        path = Path(f"runs/m11/metrics_students_{p}.json")
        if not path.exists():
            continue
        data, groups = load(path)
        for g in groups.values():
            add_best_classical(g)
        key = f"distill_host_{p}"
        for g in ("in_distribution", "tasks"):
            if g not in groups or key not in groups[g]:
                continue
            for r in sorted({m["regime"] for m in groups[g][key]}, key=lambda r: ("hold", "free", "policy", "step", "contact2", "blocked", "task", "push", "pick").index(r)):
                bc = mean_metric(groups[g]["best_classical"], r, "composite_cap"); st = mean_metric(groups[g][key], r, "composite_cap"); orc = mean_metric(groups[g]["oracle_pd"], r, "composite_cap")
                rows.append([PROFILE_LABELS[p], r, f"{bc:.2f}", f"{orc:.2f}", f"{st:.2f} ({100 * (st / bc - 1):+.0f}%)", f"{1e3 * mean_metric(groups[g][key], r, 'pos_rmse'):.1f} / {1e3 * mean_metric(groups[g]['best_classical'], r, 'pos_rmse'):.1f}",
                             f"{mean_metric(groups[g][key], r, 'excess_force_cap'):.1f} / {mean_metric(groups[g]['best_classical'], r, 'excess_force_cap'):.1f}", f"{mean_metric(groups[g][key], r, 'torque_rate_rms'):.0f} / {mean_metric(groups[g]['best_classical'], r, 'torque_rate_rms'):.0f}"])
    return table(["Profile", "Family", "Best classical", "Oracle", "Student (vs best classical)", "Position RMSE, student / classical (mrad)", "Excess force above the cap (N)", "Torque rate (Nm/s)"], rows,
                 caption="The host-rate student through each profile against the best classical controller of the family and the oracle through the same profile (cap composites, 16 cases per family).", css_class="wide") if rows else ""


ACROSS_SECTION = "<h2>The students across the profiles</h2>\n" + _across_profiles()
CRITERION_SECTION = ("<h2>The success criterion</h2>\n" + criterion_table(_groups, LEARNED)) if _groups and LEARNED else ""
WINS_SECTION = ("<h2>Where the student wins and loses</h2>\n" + wins_table(_data, METHODS, "best_classical")) if _data and LEARNED else ""
SMOOTH_SECTION = ("<h2>Smoothness</h2>\n" + torque_rate_table(_groups, [k for k, _ in METHODS])) if _groups else ""
_probes = {p: Path(f"runs/m11/probe_student_{p}.json") for p in ("p1", "p2", "p3")}
PROBE_SECTION = "<h2>What the deeper representations encode</h2>\n" + "\n".join(probe_table(path, caption=f"Probes of the host-rate student through {PROFILE_LABELS[p]}, layer by layer (four layers of 256).") for p, path in _probes.items() if path.exists())
_probe_ext = Path("runs/m11/probe_ext_student_p1.json")
PROBE_SECTION += ("\n<p>Extended targets (ADR 0011 amendment 3, item 6) on the P1 student: a nonlinear probe beside the linear one, the reference's depth inside the surface, sliding against pressing, the phases of a blocked episode, and the block in the grasp tolerance before the grasp.</p>\n" + probe_table(_probe_ext, extended=True)) if _probe_ext.exists() else ""
_variants = variants_table(["runs/m11/metrics_phase2e_p1.json", "runs/m11/metrics_phase2c_p1.json"])
VARIANTS_SECTION = ("<h2>The blocked family at the host rate</h2>\n<p>The P1 student misses the blocked family where the YAM student took it. Three diagnostics without training, then the recipe varied (ADR 0011 amendment 3, item 2). The label fit along the student's own episodes, with the torque error projected on the contact normal; the blocked phase in time, which says whether the excess is the impact transient or the steady press; and the variants: contact-weighted labels, the teacher's share, and a label term in force space.</p>\n"
                    + diag_table() + transient_table() + _variants) if (_variants or diag_table()) else ""

NARRATIVE = {
    "question": """<p>Phase 2b asks whether the learned controller of reports 15 to 18 can be rebuilt natively on a position interface: a student that decides once per host frame on the quantised feedback, whose fast law is the servo's own loop, distilled from scratch through the profile by the two-teacher recipe, with a deeper trunk (four layers of 256) and no compute latency, as the advisor allowed for this round. The criterion is the one of ADR 0010 through the interface: within 5% of the best classical controller of each family through the same profile, or better, with the oracle through the profile as the ceiling.</p>""",
    "setup": """<p><strong>Students.</strong> One per profile (P1 at 250 Hz, P2 at 100 Hz, P3 at 50 Hz), gain-residual form on the feedforward PD base with the tracking-tuned gains, the external-torque estimate in the history, a history of 100 ms at the host rate (25, 10 and 5 frames), four layers of 256; distilled by DAgger from the ideal-robot labels in free space and the oracle around the feedforward PD in contact (7 N floor), twelve iterations of 48 episodes, the teacher driving in the first two, free-space labels weighted three times; every episode runs through the profile's servo loop with the model-based interface, the student's frame mapped like any other.</p>
<p><strong>Benchmark.</strong> Protocol v4 through the profile: every group of protocol v3, the task families with the grasp, and the torque-tracking family; the classical set and the oracle through the same profile; the student's trunk probed layer by layer as in report 19.</p>""",
    "extra_sections": [s for s in (ACROSS_SECTION, CRITERION_SECTION, WINS_SECTION, SMOOTH_SECTION, VARIANTS_SECTION, PROBE_SECTION) if s.strip() and not s.endswith("</h2>\n")],
    "findings": """<ol>
<li><strong>Through every profile the host-rate student rebuilds most of the Panda controller and misses the blocked family.</strong> Distilled from scratch (best validation at the eleventh, eleventh and ninth iteration through P1, P2 and P3; 3.45, 3.73 and 4.04 against 4.21, 4.42 and 4.58 for the feedforward PD through the same profile), the students meet 25, 24 and 26 of 33 rows against the torque-interface classical set through the profile, with no divergence in any group; with the PD re-tuned through each profile added to that set (amendment 3, item 1), 21, 22 and 18. They win hold on every profile (1.3 to 1.5 against 8.2 mrad), steps (2.5 to 2.7 against 3.3 to 3.6), the task family (3.4 to 3.6 against 4.4 to 4.8, better than the oracle through P1 and P3), press-and-slide on P1 and P3, and the pick: 16, 16 and 14 of 16 grasps against 7, with the place force unchanged. Free motion is within 3% on P1 and P3 and 7% on P2 (2.1 to 4.3 against 5.0 to 6.4 mrad, at a higher torque rate), the policy-like targets within 4 to 8%. The misses are the blocked family in every group but one (11.7, 8.8 and 11.5 against 10.6 in distribution, where the oracle through the profile is at 7.4, 5.2 and 5.5; 84, 36 and 49 N above the cap with the 3 kg payload against 43), free motion and policy-like targets at the two speeds through the torque rate (P2 +17% to +33%, P3 +7% to +11%), and the push family on P2 and P3 (+23% and +17%, 7 and 3 successes against 5 and 4). Every student presses above the 7 to 12 N band on the sliding families (17 to 23 N; the feedforward PD 18): the contact impedance did not distil through the interface in twelve iterations as it did at 1 kHz on the Panda, while the ideal-robot labels in free space did. The YAM student of report 20 took the blocked family with the same recipe, so the Panda's stiffer gains and its 15 N cap against the YAM's 12 N on a quarter of the torque are the likely difference.</li>
<li><strong>Erratum (2026-11-17): no collection episode diverged through any profile.</strong> The first version of this report read 5 to 9 flagged episodes per iteration through P2 and up to 28 of 48 through P3 as divergences of the student-driven collection and offered a mechanism (the student's stiffness folded into a held goal). The flag was an artefact: the free-space labelling path marked an episode as diverged when it held fewer labelled decisions than a tenth of the physics steps, a threshold written for decisions at 1 kHz. Through P3 a 5 s episode has 250 decisions against a threshold of 500, so every free-space episode tripped it (the flagged count equals the number of free-space episodes drawn in each of the twelve iterations), and through P2 the decisions dropped at saturation tipped some of the 500 over. No training episode blew up through any profile, the labels were aggregated regardless, and the students are unaffected; the smaller label stores of P2 and P3 (248 000 and 134 000 rows against 300 000) follow from the host rate alone. The flag now counts decisions and reports a blow-up separately.</li>
<li><strong>Depth and the host rate buy a sharper contact representation.</strong> Probed layer by layer through their profiles, the four-layer students decode contact at 0.84 to 0.87 (P1), 0.89 to 0.94 (P2) and 0.85 to 0.90 (P3) balanced accuracy, detecting 54 to 88% of onsets within 50 ms, against 0.77 to 0.80 and 30 to 44% for the two-layer Panda student of report 19; the surface normal is readable from their observations (R² 0.68 to 0.76) and from no layer, the stiffness classes stay at chance, the payload and the contact force are not linearly decodable.</li>
<li><strong>The blocked miss is the impact transient under payload, not the label fit.</strong> Along the P1 student's own episodes the teacher's labels are fitted to 2.4 Nm RMS on a 16 Nm RMS yield in the blocked phase (correlation 0.98; 0.6 Nm on the approach), and the torque error projected on the contact normal is unbiased (a push error of 0.4 N mean, 7 N RMS): the regression is not short of the contact labels, and the student yields with its gain head (kp multipliers 0.59 inside the surface) as the oracle does. In time, 54% of the student's excess above the cap falls in the first half second after the nominal contact (the oracle's 31%), its peak is 185 N against the oracle's 102, and its steady press in the last half second is 16.3 N against 14.0; by payload its excess is 7.0, 11.9 and 33.8 N at 0, 1 and 2 kg against the oracle's 7.9, 8.2 and 11.4. The oracle is privileged with the payload and compensates its weight along the normal; the student has to infer it, and the probes find the payload not linearly decodable from any layer (finding 6). On the YAM, whose payloads reach 1 kg on a 12 N cap, the student's excess by payload (2.5, 5.0, 7.8 N) stays under the oracle's (0.9, 5.3, 12.7).</li>
<li><strong>Of the recipe's variants, keeping the teacher in the loop helps and re-weighting the labels does not.</strong> Weighting the contact rows three times (free 1, contact 3) made the blocked family worse at every iteration (13 to 23 on validation) and the selection rule returned the warm start, that is, the feedforward PD; neutral weights (1, 1) selected the seventh iteration at 3.79 against the recipe's 3.45 and pressed harder on the slides (22 N). The teacher's share held at 0.5 for three more iterations selected the eleventh iteration at 3.28: blocked 8.28 against the recipe's 11.74 (the oracle through P1 7.41; excess 19.7 against 23.5 N) with hold and free motion kept (0.92, 0.84 against 0.86, 0.86) and the press on the sliding family worse (21.9 against 17.0 N, the band 7 to 12). The force-space label term (the force along the normal the torque error implies, per the cap, at decisions in contact) at weight 1 dominated the regression (its loss 0.02 to 0.1 against 0.002 for the torque term) and never validated below the feedforward PD, so its selected checkpoint is the warm start; its evaluation was dropped. With the PD re-tuned through the profile in the classical set, the recipe's student meets 21 of 33 rows (it had 25 against the torque-interface gains alone: the re-tuned PD takes the blocked and policy-like families and the fast free motion), and the teacher's-share variant 25 of 33: every blocked row in every group (6 to 15% better than the best classical controller, the re-tuned PD or the cap-tuned one), hold, steps, the task family and the pick; its misses are the policy-like family (6 to 19% worse than the re-tuned PD through its torque rate), free motion at the two speeds (21 and 31%), and the push (0.78 against 0.73). Against the pre-registered target of amendment 3, blocked excess within 5% of the oracle's, it falls short (19.7 against 12.7 N); on the family's criterion rows it is the first host-rate student on the Panda to take the blocked family. Seeds 1 and 2 of the teacher's-share recipe select their sixth and twelfth iteration at 4.04 and 3.24 on validation: seed 2 repeats seed 0 (blocked 10.14, excess 27.2 N, hold 0.85, free 0.83), seed 1 does not (blocked 11.27, as the recipe; hold 0.69), so pooled over the three seeds the variant meets 23 of 33 rows, taking blocked in distribution (9.90 against the re-tuned PD's 9.68) and on the compliant 500 N/m surface and losing it under the 3 kg payload (+35%), on the out-of-distribution instances (+19%) and on the 2 kN/m surface (+6%), besides the policy-like family (+8 to +22%) and the fast free motion (+20 and +28%); every seed grasps 16 of 16. The between-seed spread on blocked (8.3 to 11.3) is the size of the recipe's effect, so the blocked family at the host rate is improved on average and not settled. The same recipe with eight layers of 256 never validated below the feedforward PD in twelve iterations (its selected checkpoint is the warm start): with this budget the deeper trunk does not train from scratch, and decision 6's depth comparison stands at four layers against two.</li>
</ol>""",
    "caveats": """<ul>
<li>One seed per profile and per recipe variant (the teacher's-share recipe has seeds 1 and 2 and an eight-layer form, finding 5); the force-space term was run at one weight; the students are distilled from scratch, without the actor-critic base of the Panda students, so their free-space part rests on the ideal-robot labels alone.</li>
<li>The diagnosis tables and the transient split are from eight train cases per family and the sixteen in-distribution blocked test cases respectively, one student; the push error along the normal is the least-squares force through the tool Jacobian at the measured configuration.</li>
<li>The oracle and the classical set are through the same profile, so the comparison is fair to the interface but every ceiling is itself interface-limited.</li>
</ul>""",
    "next": """<ul>
<li>The blocked family under payload is the open item: the student's excess is the impact transient when it carries 1 to 2 kg, where the oracle's privilege is the payload's weight along the normal. A payload estimate as an explicit input, or labels that make the student infer it, is the next intervention; the teacher's share in the collection is the recipe change that worked.</li>
<li>The end-to-end control arm of ADR 0011 decision 5 (a student that emits the servo frame, fitted through the servo's law over the hold) was not run in this round.</li>
<li>Seeds 1 and 2 at P1 on the recipe that comes out of the blocked-family work (the advisor's decision of 2026-11-17: contact-weighted labels; no further seeds on the YAM; the Koch arm deferred).</li>
</ul>""",
}
