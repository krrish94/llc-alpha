"""Checkpoint 2 (Milestone 10, ADR 0010): a standalone summary of the across-the-board benchmark on
plant v3, in the style of Checkpoint 1: the motivation, the plant and the protocol, the learned
controller against the best classical controller of every family, where it still loses, and what
was learned about training.

    LLC_PLANT=v3 uv run python experiments/checkpoint_2.py --metrics runs/m10r2/metrics_test.json --episodes runs/m10r2/episodes_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, method_color, reference_color  # noqa: E402
from report_api import compare_stats, figure, table  # noqa: E402

from experiments.d7_extras import label, mean_metric  # noqa: E402
from experiments.d_report import GROUP_LABELS, REGIME_SHORT, add_best_classical, build_compare, pooled, wins_table  # noqa: E402
from experiments.m10_extras import criterion_table, latency_table, torque_rate_table  # noqa: E402
import experiments.m10_narrative  # noqa: E402,F401  (updates the method labels of the Milestone 10 controllers)

REPORT = Path("research/reports/16-checkpoint-2")
FIG = REPORT / "figures"
BEST = "best_classical"
from experiments.d7_extras import LABELS as _LABELS  # noqa: E402

_LABELS.update({"distill_v3_r2_two_teachers": "Learned controller (two-teacher student)", "distill_v3_r2_gain_ext": "Round-2 student, oracle teacher only",
                "distill_v3_gain_ext": "Milestone 10 student (gravity base)", "distill_v3_r2_torque": "Student of the direct torque network", "pd_residual_v3_ff": "Residual on the feedforward PD (base)"})
FAMILIES = ("hold", "free", "policy", "step", "contact2", "blocked", "task")


def load(path: Path):
    d = json.loads(path.read_text())
    groups = {g: add_best_classical(pooled(res, d["learned"])[0]) for g, res in d["groups"].items()}
    return groups, d


def paired(groups: dict, gname: str, methods: list[str], regime: str, key: str) -> dict:
    cmp = build_compare(groups[gname], [(m, m) for m in methods], BEST)
    return {s["key"]: s for s in compare_stats(cmp, regime, key)}


def fig_hero(groups: dict, methods: list[str]) -> str:
    """Relative change of the learned controllers against the best classical controller on the headline metrics."""
    apply_style()
    spec = [("hold", "pos_rmse", "Hold: position RMSE"), ("free", "pos_rmse", "Free motion: position RMSE"), ("policy", "pos_rmse", "Policy-like: position RMSE"),
            ("step", "settle_time", "Step: settling time"), ("contact2", "excess_force_cap", "Press and slide: force above 15 N"), ("contact2", "press_deficit", "Press and slide: press deficit"),
            ("blocked", "excess_force_cap", "Blocked: force above 15 N"), ("blocked", "blocked_drift", "Blocked: drift from the contact point"), ("blocked", "release_reaction", "Blocked: release reaction"),
            ("task", "excess_force_cap", "Task: force above 15 N"), ("push", "push_error", "Push: distance error"), ("pick", "place_force_peak_20ms", "Place: force on the table")]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    y = np.arange(len(spec))[::-1]
    h = 0.8 / max(1, len(methods))
    for j, mth in enumerate(methods):
        vals, sig = [], []
        for regime, key, _ in spec:
            gname = "tasks" if regime in ("push", "pick") else "in_distribution"
            st = paired(groups, gname, [BEST] + methods, regime, key)
            ref, s = st[BEST]["mean"], st.get(mth)
            if s is None or not np.isfinite(s["mean"]):
                vals.append(0.0); sig.append(False); continue
            vals.append((s["mean"] - ref) / (abs(ref) + 1e-9) * 100)
            lo, hi = s["diff_ci"]; sig.append(not (np.isnan(lo) or lo <= 0 <= hi))
        offs = y + (len(methods) - 1) * h / 2 - j * h
        ax.barh(offs, vals, height=h * 0.9, color=method_color(mth), alpha=0.95, label=label(mth))
        for yy, v, sg in zip(offs, vals, sig):
            ax.text(min(v, 0) - 3 if v < 0 else v + 3, yy, f"{v:+.0f}%" + ("" if sg else " (n.s.)"), va="center", ha="right" if v < 0 else "left", fontsize=7)
    ax.axvline(0, color=reference_color(), lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([s[2] for s in spec])
    ax.set_xlabel("change against the best classical controller of the family (%; negative is better for every metric shown)")
    ax.set_xscale("symlog", linthresh=100, linscale=1.2); ax.set_xlim(-110, 1500)
    ax.set_xticks([-100, -50, 0, 50, 100, 300, 1000]); ax.set_xticklabels(["-100%", "-50%", "0", "+50%", "+100%", "+300%", "+1000%"])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG / "hero.svg", bbox_inches="tight"); plt.close(fig)
    return figure("figures/hero.svg", "Relative change of the learned controllers against the best classical controller of each family (the lower cap composite among the two tuned PDs and their feedforward variants; 16 cases per family, seeds pooled). Every metric shown is lower-is-better. Not significant (n.s.): the 95% bootstrap interval of the paired difference includes zero. Symmetric log axis beyond 50%.", "fig-hero", wide=True)


def fig_groups(groups: dict, methods: list[str]) -> str:
    apply_style()
    names = [g for g in groups if g != "in_distribution" and not g.startswith("latency")]
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    x = np.arange(len(names)); w = 0.8 / (len(methods) + 1)
    for j, mk in enumerate([BEST] + methods):
        vals = [np.mean([mean_metric(groups[g][mk], r, "composite_cap") for r in sorted({m["regime"] for m in groups[g][mk]})]) if mk in groups[g] else np.nan for g in names]
        ax.bar(x + (j - len(methods) / 2) * w, vals, width=w * 0.95, color=method_color(mk), label=label(mk))
    ax.set_xticks(x); ax.set_xticklabels([GROUP_LABELS.get(g, g).split(" (")[0] for g in names], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("mean cap composite over the group's families"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "groups.svg"); plt.close(fig)
    return figure("figures/groups.svg", "Held-out groups, compliant surfaces and the task families: mean cap composite over the families of each group (lower is better), the best classical controller against the learned ones.", "fig-groups", wide=True)


def _load_ep(root: Path, case: str, method: str):
    p = root / f"{case}__{method}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def fig_traces(root: Path, reps: dict, methods: list[tuple[str, str]]) -> str:
    apply_style()
    cases = [(reps.get("contact2"), "Press and slide", "force"), (reps.get("blocked"), "Blocked target", "force"), (reps.get("task"), "Task", "force"),
             (reps.get("hold"), "Hold with a 10 N pulse", "error"), (reps.get("step"), "Step targets", "error"), (reps.get("free"), "Free motion", "error")]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.6))
    for ax, (case, title, kind) in zip(axes.flat, cases):
        for key, fname in methods:
            d = _load_ep(root, case, fname) if case else None
            if d is None:
                continue
            t = d["t"]
            if kind == "force":
                ax.plot(t, d["f_normal"], color=method_color(key), lw=0.9, label=label(key)); ax.axhline(15.0, color=reference_color(), lw=0.7, ls=(0, (4, 2))); ax.set_ylabel("normal contact force (N)")
            else:
                ax.plot(t, 1e3 * np.abs(d["q_ref"] - d["q"]).max(1), color=method_color(key), lw=0.9, label=label(key)); ax.set_ylabel("max joint error (mrad)")
        ax.set_title(f"{title} ({case})", fontsize=9); ax.set_xlabel("time (s)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / "traces.svg", bbox_inches="tight"); plt.close(fig)
    return figure("figures/traces.svg", "Representative test cases (the median case of the best classical controller per family): contact force in the three contact families, maximum joint error in three free-space families.", "fig-traces", wide=True)


def protocol_table() -> str:
    rows = [["Hold", "Hold a configuration; a 10 N tool pulse of 0.2 s at 1 s", "position RMSE, pulse deflection, torque rate", "small error, small deflection, smooth torque"],
            ["Free motion", "Minimum-jerk joint paths between random configurations", "position RMSE, torque rate", "accurate, smooth"],
            ["Policy-like targets", "Held targets at 10 to 50 Hz with jitter and jumps", "position RMSE, torque rate", "track the intent, no chatter"],
            ["Step targets", "Joint-space steps", "position RMSE, settling time, overshoot", "fast settling, no overshoot"],
            ["Press and slide", "Approach, press 0.4 to 1.4 cm (7 to 12 N), slide, release, on the table or the wall", "force above 15 N, sustained peak, press deficit (below 7 N), contact losses, tool error", "press in the band, keep contact, no impact"],
            ["Blocked target", "A path through the table or the wall (3 to 8 cm)", "force above 15 N, sustained peak, drift from the contact point, fraction moving, release reaction and settling", "stop at the surface and wait, resume when the target returns"],
            ["Task (held out)", "Reach, press 1 cm, slide 10 cm, lift, transport, press 0.5 cm, lift", "force above 15 N, sustained peak, press deficit, contact losses", "the press band under a task-like sequence"],
            ["Box pushing", "Push a 5 cm box 10 cm with the fingertip faces", "success (errors below 2 cm, yaw below 15 degrees), force on the box", "move the box where asked, gently"],
            ["Pick-and-place", "Carry a held block, place it on the table, release, lift", "success (error below 2 cm, upright), block-table force, lift tracking", "place gently, track after the payload change"]]
    return table(["Family", "Reference", "Measured", "Wanted"], rows, caption="Protocol v3 on plant v3 (ADR 0009): 16 test cases per family, each with a robot instance, a payload of 0 to 2 kg and sensor noise; held-out groups: workspace, speed x1.5 and x2, payload 3 kg, doubled instance deviations, compliant surfaces of 2 kN/m and 500 N/m; a 5 ms compute-latency group for the learned controllers.", css_class="wide")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/m10r2/metrics_test.json")
    ap.add_argument("--episodes", default="runs/m10r2/episodes_test")
    ap.add_argument("--learned", nargs="*", default=["distill_v3_r2_two_teachers", "distill_v3_r2_gain_ext", "distill_v3_gain_ext", "distill_v3_r2_torque"])
    args = ap.parse_args()
    groups, data = load(Path(args.metrics))
    learned = [k for k in args.learned if k in groups["in_distribution"]]
    methods = [(BEST, BEST)] + [(k, k) for k in learned]
    reps = data.get("representative_cases", {})
    src = {m["regime"]: m.get("best_classical_source") for m in groups["in_distribution"][BEST]}
    trace_methods = [(BEST, src.get("blocked", "pd_shared"))] + [(k, f"{next(n for n, info in data['learned'].items() if (info.get('method_key') or '') == k)}") for k in learned if any((info.get("method_key") or "") == k for info in data["learned"].values())]
    parts = [f"<h2>Summary</h2>\n{SUMMARY}", "<h2>1. Motivation</h2>\n" + MOTIVATION, "<h2>2. Plant and controllers</h2>\n" + CONTROLLERS, "<h2>3. Protocol</h2>\n" + protocol_table(),
             "<h2>4. The evidence</h2>\n" + fig_hero(groups, learned) + criterion_table(groups, learned) + fig_groups(groups, learned) + fig_traces(Path(args.episodes), reps, trace_methods),
             "<h2>5. Where it still loses</h2>\n" + wins_table(data, methods, BEST, caption="Every scenario and metric with a significant paired difference against the best classical controller of the family (95% bootstrap interval on per-case differences excludes zero); parity rows omitted."),
             "<h2>6. Realism checks</h2>\n" + latency_table(groups, learned) + torque_rate_table(groups, [BEST] + learned)]
    videos = REPORT / "data" / "videos.html"
    if videos.exists():
        parts.append("<h2>7. Representative episodes</h2>\n<p>One case per family (the median case of the tracking-tuned PD), the two tuned PD variants, the privileged oracle and the distilled student side by side; the contact families also at quarter speed.</p>\n" + videos.read_text())
    parts += ["<h2>8. Training lessons</h2>\n" + LESSONS, "<h2>9. Next</h2>\n" + NEXT]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report.html").write_text("\n\n".join(parts))
    print("wrote", REPORT / "report.html")


SUMMARY = """<p><strong>Verdict (revised after round 2, report 17).</strong> One learned controller, a residual-plus-gain-multiplier network on a 1 kHz PD base that carries the nominal inverse-dynamics feedforward of the reference, with proprioception and the external-torque estimate as inputs, trained by backpropagation through the plant on the free-space families and then distilled from two privileged teachers (the ideal-robot torque in free space, the contact-impedance oracle in contact) with its free space protected, is competitive or better than the best classical controller of each family on 28 of the 33 family-and-group rows of protocol v3 on the Panda with a Robotiq 2F-85. It tracks free motion at 2.6 mrad against 4.9 for the feedforward PD (the first-round controller, without the model term, was at 11.9), hold at 1.6 against 8.1 mrad, and keeps that advantage in every held-out group (5.0 against 20.6 mrad with a 3 kg payload); it wins the three contact families by 13 to 52% (stopping and waiting at a blocked surface with 5 N above the 15 N cap where the best classical pushes 23 N, pressing inside the 7 to 12 N band where the first-round controller pressed below it), every held-out group's contact families, and keeps a 5 ms compute latency within 5%. The five misses are all the policy-like family (20 Hz held targets), at 5 to 6% above the best classical in every group, with a lower tracking error and a higher torque rate. Push success is 52% against 31% for every classical controller, place success 100% for all with placement at 5 N against 14 N. The pre-registered criterion of ADR 0010 is therefore met on every family but the policy-like targets, by a margin at the threshold, and on the task-success half it is met on pushing and tied at the ceiling on placing. The same student distilled from the oracle alone meets 29 rows and misses free motion where the nominal model is accurate (6.9 against 4.9 mrad); a direct torque network as the learned part fails this protocol: its student loses every free-space family and degrades threefold under latency, because a torque held for 10 ms is open-loop for 10 ms.</p>"""
MOTIVATION = """<p>Every torque-controlled robot runs a low-level feedback loop between the policy's targets and the motors, usually a PD or impedance law at 100 Hz to 1 kHz with hand-tuned gains. The research agenda argues for learning this loop: no single gain set serves every task, phase of a task and robot instance; policies treat the loop as a black box and pay for it in jerky, unstable behaviour; retuning by hand does not scale across robots, tasks and contact modes, and switching gain sets stably is hard in dynamic tasks; and such a controller, learned from proprioception in simulation, has a small sim-to-real gap. Checkpoint 1 showed the first evidence on a Panda with a spherical tool: a learned residual tracked better than the tuned PD in every family and limited contact forces far better, but did not stop and wait at a blocked surface nor press to a target. Deep dive D6 obtained that behaviour by imitating a privileged oracle, and Milestone 9 moved the plant to the DROID robot (Panda with a Robotiq 2F-85) with a task-level protocol. This checkpoint asks the question the agenda set: is there one learned controller that is competitive or better than the best classical baseline on every family and condition at once?</p>"""
CONTROLLERS = """<p><strong>Plant.</strong> MuJoCo, 1 kHz, the Panda with the Robotiq 2F-85 baked closed into one rigid tool body (open to 38 mm for pick-and-place), a table and a wall with the explicit soft contact of ADR 0001, robot instances (mass, inertia, centre-of-mass, damping, armature, friction and actuator-gain deviations), payloads of 0 to 3 kg and sensor noise. The nominal model every controller uses is the datasheet robot.</p>
<p><strong>Classical set.</strong> Two shared PD + gravity laws tuned by CMA-ES on the training split, one on the tracking composite and one on the cap composite, and the inverse-dynamics feedforward variant of each; the best of the four per family is the reference of every comparison. A privileged oracle (the contact-impedance law of ADR 0008 around the PD, knowing the surface and the payload) marks the ceiling of the contact families.</p>
<p><strong>Learned controllers.</strong> All run at 10 ms decisions on a 1 kHz PD base with proprioception, the commanded-torque history and the external-torque estimate (link-side torque minus the model) as inputs; the round-2 controllers' PD base carries the nominal inverse-dynamics feedforward of the reference, M(q) qdd_ref + C(q, qd_ref) qd_ref + g(q), the same term the feedforward PD of the classical set has (ADR 0010, amendment 2). The residual by actor-critic through the plant (D3 recipe): a torque residual trained by backpropagation through MuJoCo with bounded tracking kernels and a privileged critic. The distilled impedance student: per-joint gain multipliers plus a residual, trained by DAgger to imitate the oracle around that residual, with the free-space families weighted up so that the residual's own behaviour survives where there is no contact. The TAM-style ideal-robot residual (Son et al., CoRL 2026, re-implemented without multi-robot pretraining): a residual trained to make the perturbed robot reproduce the nominal robot's response. The round-2 students (report 17): the residual's base trained on the free-space families only and selected there, the teacher's press floor at the protocol's 7 N, then distilled either from the oracle alone or from two teachers, the TAM-style ideal-robot torque (the torque the sampled plant needs to produce the nominal robot's response under the nominal law, privileged) in free space and the oracle in contact; the first-round student is kept in the tables as the controller they replace. The direct torque network (round 2): the same history input, a torque output instead of a residual, warm-started on the feedforward PD and trained by the same actor-critic, with its own distilled student.</p>"""
LESSONS = """<ul>
<li><strong>Specify the behaviour, then imitate it.</strong> The contact behaviour was written down as a force-bounded spring on the actual error, executed by a privileged oracle, validated against the metrics, and distilled by DAgger; reinforcement learning (PPO) never found it by exploration and did not improve it once found.</li>
<li><strong>Protect what the base already does.</strong> Distillation with the teacher driving only once and the free-space decisions weighted up keeps the base residual's free space within a few percent; the earlier recipe gave up 20 to 40% of it.</li>
<li><strong>The cue is the external torque.</strong> The link-side torque minus the robot's own model is the only proprioceptive cue that detects contact within 50 ms for more than 90% of onsets; the tool-frame wrench was shelved for its sim-to-real gap.</li>
<li><strong>Selection decides what a method looks like.</strong> The actor-critic residual's best checkpoint under the cap composite is a contact-stage iteration whose free space is half as good as its stage-1 optimum; the ideal-robot residual's best checkpoint is the PD itself. Report the rule and the alternative.</li>
<li><strong>Give the learned controller the model term, then supervise the rest.</strong> A classical inverse-dynamics feedforward halves the tracking error of the tuned PD; a residual on top of that feedforward recovers most of the gap (11.9 to 6.9 mrad against 4.9), limited by a training reward whose bounded position kernel and torque-rate penalty do not distinguish 5 from 7 mrad; ideal-robot labels in free space, referenced to the nominal law, halve the feedforward PD's error (2.6 mrad). Referenced to the student's own command, the same labels drift with the student and degrade it in every iteration.</li>
<li><strong>Keep the 1 kHz base under the learned part.</strong> A direct torque network at 10 ms decisions is the best hold tracker as a base and the worst student: held for 10 ms its torque is open-loop, it chatters (1953 Nm/s), and a 5 ms latency degrades it threefold; the residual form rides on the PD between decisions and moves by less than 3%.</li>
<li><strong>Plant transfer is real but partial.</strong> The controller trained on the sphere plant keeps its contact behaviour on the gripper and loses its free-space edge there; retraining on the new plant restores it.</li>
</ul>"""
NEXT = """<ul>
<li>The policy-like family (held targets at 20 Hz), the one miss at 5 to 6%: the ideal labels are dropped where the nominal law saturates, so the student's behaviour at target jumps is the base's; a smoothness term in the free-space labels, or the oracle's labels on that family, would settle it.</li>
<li>Pushing: 54 against 31% on sixteen cases is eight or nine boxes against five; more cases before it enters a claim. The box's yaw remains the failure for every controller.</li>
<li>The press band against the cap: the 7 N floor put the controller inside the band at the price of 5 N above the cap on the sliding families, where light contact detaches on this plant; the plant question of report 14 comes before further tuning of the teacher.</li>
<li>The real robot (Milestone 7) remains unauthorised; the controller's inputs and outputs are those a Panda exposes.</li>
</ul>"""

if __name__ == "__main__":
    main()
