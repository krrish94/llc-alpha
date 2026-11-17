"""Checkpoint 1: a standalone summary of the project so far (research/reports/12-checkpoint-1).

Written for the advisor and for readers who have not followed the deep dives: motivation, the
plant and the controllers, the evaluation protocol (what is measured and what is wanted), the first
evidence that learned low-level controllers outperform a tuned PD, where they still lose, and the
open problem of compliance in contact. Figures are computed from the corrected test tables
(runs/d5/metrics_test.json, runs/d6/metrics_checkpoint.json) and the saved representative episodes.

    uv run python experiments/checkpoint_1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, method_color, reference_color  # noqa: E402
from report_api import figure, table  # noqa: E402

from experiments.d_report import build_compare, pooled, wins_table  # noqa: E402
from llc.eval.metrics import METRIC_INFO  # noqa: E402

REPORT = Path("research/reports/12-checkpoint-1")
FIG = REPORT / "figures"
D4 = Path("runs/d5/metrics_test.json")  # corrected D4 tables: PD, the D3 residual, the PPO arms
CKPT = Path("runs/d6/metrics_checkpoint.json")  # the stop-and-wait blocked metrics for the key controllers
EPISODES = Path("runs/d5/episodes")
EPISODES_CK = Path("runs/d6/episodes_checkpoint")
AUDIT = Path("runs/d6/reward_audit.json")

PD, RES, PPO_C = "pd_shared", "pd_residual_acmed_s02_lowrate", "ppo_contact_v2"
LABEL = {PD: "PD, shared gains", RES: "PD + learned residual", PPO_C: "PPO, contact families", "ppo_cap_gain": "PPO under the 15 N cap (gain residual)", "ppo_cap_torque_ext": "PPO, external-torque input"}
FAMILY = {"hold": "Hold, 10 N pulses", "free": "Free motion", "policy": "Policy-like targets", "step": "Step targets", "contact2": "Press and slide", "blocked": "Blocked target", "task": "Task (held out)"}
GROUP = {"in_distribution": "In-distribution", "heldout_workspace": "Held-out workspace", "speed_x1.5": "Speed x1.5", "speed_x2": "Speed x2", "payload_3kg": "Payload 3 kg", "ood_instances": "Held-out robot instances"}


def load_pooled(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {g: pooled(res, d["learned"])[0] for g, res in d["groups"].items()}, d


def mean_metric(res: list[dict], regime: str, key: str) -> float:
    v = [m[key] for m in res if m["regime"] == regime and np.isfinite(m.get(key, float("nan")))]
    return float(np.mean(v)) if v else float("nan")


def paired_ci(groups: dict, gname: str, methods: list[str], regime: str, key: str) -> dict:
    """Mean per method and the bootstrap CI of the paired difference against the PD (report_api)."""
    from report_api import compare_stats

    cmp = build_compare(groups[gname], [(m, m) for m in methods], PD)
    return {s["key"]: s for s in compare_stats(cmp, regime, key)}


# ------------------------------------------------------------------------------------------ figures
def fig_hero(groups: dict) -> str:
    """Relative change against the PD on the in-distribution families for the headline metrics."""
    apply_style()
    spec = [("hold", "pos_rmse", "Hold: position RMSE"), ("free", "pos_rmse", "Free motion: position RMSE"), ("policy", "pos_rmse", "Policy-like: position RMSE"),
            ("step", "settle_time", "Step: settling time"), ("contact2", "f_peak_20ms", "Press and slide: sustained peak force"), ("contact2", "excess_force_cap", "Press and slide: force above 15 N"),
            ("task", "f_peak_20ms", "Task: sustained peak force"), ("task", "excess_force_cap", "Task: force above 15 N"), ("blocked", "excess_force_cap", "Blocked: force above 15 N"),
            ("blocked", "contact_recovery", "Blocked: recovery after release"), ("task", "press_deficit", "Task: press deficit"), ("contact2", "contact_loss_rate", "Press and slide: contact losses")]
    methods = [RES, PPO_C]
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    y = np.arange(len(spec))[::-1]
    h = 0.36
    for j, mth in enumerate(methods):
        vals, sig = [], []
        for regime, key, _ in spec:
            st = paired_ci(groups, "in_distribution", [PD] + methods, regime, key)
            ref, s = st[PD]["mean"], st[mth]
            rel = (s["mean"] - ref) / (abs(ref) + 1e-9) * 100
            lo, hi = s["diff_ci"]
            vals.append(rel); sig.append(not (np.isnan(lo) or lo <= 0 <= hi))
        offs = y + (h / 2 if j == 0 else -h / 2)
        ax.barh(offs, vals, height=h, color=method_color(mth), alpha=0.95, label=LABEL[mth])
        for yy, v, sg in zip(offs, vals, sig):
            ax.text(min(v, 0) - 3 if v < 0 else v + 3, yy, f"{v:+.0f}%" + ("" if sg else " (n.s.)"), va="center", ha="right" if v < 0 else "left", fontsize=7.5)
    ax.axvline(0, color=reference_color(), lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([s[2] for s in spec])
    ax.set_xlabel("change against the tuned PD (%; negative is better for every metric shown)")
    ax.set_xscale("symlog", linthresh=100, linscale=1.2)
    ax.set_xlim(-110, 1500)
    ax.set_xticks([-100, -50, 0, 50, 100, 300, 1000]); ax.set_xticklabels(["-100%", "-50%", "0", "+50%", "+100%", "+300%", "+1000%"])
    ax.legend(loc="upper right")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "hero.svg"); plt.close(fig)
    return figure("figures/hero.svg", "Relative change of two learned controllers against the tuned PD on the in-distribution families (16 cases per family; mean over cases). Every metric shown is lower-is-better, so bars to the left are improvements. The first eight rows are the headline wins (tracking and force limitation); the last four are where learned controllers lose (pressing to a target, keeping contact, recovering after a block). Not significant (n.s.): the 95% bootstrap interval of the paired difference includes zero. Symmetric log axis beyond 50%.", "fig-hero", wide=True)


def _load_ep(case: str, method: str, root: Path = EPISODES):
    p = root / f"{case}__{method}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def fig_traces() -> str:
    apply_style()
    cases = [("contact2-test-002", "Press and slide", "force"), ("task-test-013", "Task: press, slide, lift, transport", "force"), ("blocked-test-013", "Blocked target", "force"),
             ("hold-test-003", "Hold with a 10 N pulse", "error"), ("step-test-011", "Step targets", "error"), ("free-test-002", "Free motion", "error")]
    methods = [(PD, "pd_shared"), (RES, "pd_residual_acmed_s02_lowrate-s0"), (PPO_C, "ppo_contact_v2-s0")]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.6))
    for ax, (case, title, kind) in zip(axes.flat, cases):
        for key, fname in methods:
            d = _load_ep(case, fname)
            if d is None:
                continue
            t = d["t"]
            if kind == "force":
                ax.plot(t, d["f_normal"], color=method_color(key), lw=0.9, label=LABEL[key])
                ax.axhline(15.0, color=reference_color(), lw=0.7, ls=(0, (4, 2)))
                ax.set_ylabel("normal contact force (N)")
            else:
                err = np.abs(d["q"] - d["q_ref"]).max(1) * 1e3
                ax.plot(t, err, color=method_color(key), lw=0.9, label=LABEL[key])
                ax.set_ylabel("largest joint error (mrad)")
        ax.set_title(title); ax.set_xlabel("time (s)")
    axes[0, 0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "traces.svg"); plt.close(fig)
    return figure("figures/traces.svg", "Representative episodes (median composite case of the PD per family). Top: normal contact force; the dashed line is the 15 N cap. Bottom: largest joint tracking error. The learned controllers keep the contact force low on the press-and-slide and task cases and track more tightly in free space; on the blocked target all three push hard against the surface.", "fig-traces", wide=True)


def fig_groups(groups: dict) -> str:
    """Generalisation: the same two metrics across the held-out groups."""
    apply_style()
    gnames = [g for g in ("in_distribution", "heldout_workspace", "speed_x1.5", "speed_x2", "payload_3kg", "ood_instances") if g in groups]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for ax, (key, regimes, ylabel) in zip(axes, [("pos_rmse", ("free", "policy"), "position RMSE, free and policy-like (rad)"), ("f_peak_20ms", ("contact2", "task"), "sustained peak force, press-slide and task (N)")]):
        x = np.arange(len(gnames)); w = 0.26
        for j, mth in enumerate((PD, RES, PPO_C)):
            vals = []
            for g in gnames:
                v = [m[key] for m in groups[g].get(mth, []) if m["regime"] in regimes and np.isfinite(m.get(key, float("nan")))]
                vals.append(float(np.mean(v)) if v else np.nan)
            ax.bar(x + (j - 1) * w, vals, width=w, color=method_color(mth), label=LABEL[mth])
        ax.set_xticks(x); ax.set_xticklabels([GROUP[g] for g in gnames], rotation=20, ha="right"); ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "groups.svg"); plt.close(fig)
    return figure("figures/groups.svg", "The same advantage under every held-out condition: tracking error in free motion (left) and sustained peak contact force (right) for the tuned PD and two learned controllers, in distribution and under a held-out workspace, faster references, a 3 kg payload and robot instances with doubled parameter deviations. None of these conditions was used for tuning or selection.", "fig-groups", wide=True)


def fig_blocked(ck_groups: dict | None) -> str:
    """Stop at the surface and wait: drift from the first-contact point and force while blocked."""
    if ck_groups is None:
        return ""
    apply_style()
    res = ck_groups["in_distribution"]
    methods = [m for m in (PD, RES, "ppo_cap_gain", "ppo_cap_torque_ext") if m in res]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    for ax, (key, ylabel) in zip(axes, [("blocked_drift", "tangential drift from the contact point (cm)"), ("excess_force_cap", "force above the 15 N cap (N)"), ("contact_recovery", "recovery after release (s)")]):
        vals = [mean_metric(res[m], "blocked", key) * (100 if key == "blocked_drift" else 1) for m in methods]
        ax.bar(range(len(methods)), vals, color=[method_color(m) for m in methods])
        ax.set_xticks(range(len(methods))); ax.set_xticklabels([LABEL.get(m, m) for m in methods], rotation=15, ha="right", fontsize=8); ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(FIG / "blocked.svg"); plt.close(fig)
    return figure("figures/blocked.svg", "The blocked target under the new definition 'stop at the surface and wait': mean tangential drift of the tool from the point where it first touched (lower is better), force above the 15 N cap, and recovery after the reference returns; 16 blocked cases, in distribution. No controller yet waits with a capped force: all push with 45 to 60 N above the cap.", "fig-blocked", wide=True)


# ------------------------------------------------------------------------------------------ tables
def protocol_table() -> str:
    rows = [
        ["Hold with pulses", "hold a fixed posture; a 10 N force pulse of 0.2 s hits the tool", "position RMSE, deflection under the pulse, torque-rate RMS", "small error and deflection, smooth torques"],
        ["Free motion", "smooth joint-space motions through the workspace", "position RMSE, torque-rate RMS", "tight tracking, smooth torques"],
        ["Policy-like targets", "piecewise-constant targets changing every 20 ms, as a learned policy emits them", "position RMSE, torque-rate RMS", "track the staircase without excess chatter"],
        ["Step targets", "joint-space steps", "position RMSE, settling time, overshoot", "fast settling without overshoot"],
        ["Press and slide", "approach a table or wall at speed, press 0.5 to 3 cm into it, slide, release", "excess force above 15 N, sustained peak (20 ms), feasible-direction error, press deficit below 5 N, contact losses per second", "bounded forces, a maintained press, tracking along the surface"],
        ["Blocked target", "the target passes 3 to 8 cm through a table or wall and comes back", "excess force above 15 N, sustained peak, tangential drift from the contact point, motion while blocked, apparent stiffness, recovery after release", "stop at the surface and wait with a capped force, resume promptly"],
        ["Task (held out)", "reach, descend, press 1 cm, slide 10 cm, lift, transport, press 0.5 cm, lift; never trained on", "excess force, sustained peak, feasible-direction error, press deficit", "the contact behaviour of the two families above, in sequence"],
    ]
    return table(["Family", "Reference", "Primary metrics", "Wanted behaviour"], rows, caption="Protocol v2 families: 16 test cases per family drawn from a split disjoint from training and validation; every case has its own robot instance (link masses, inertias, friction and damping perturbed), payload (0 to 2 kg) and sensor noise. Held-out groups repeat the families under a shifted workspace, references 1.5 and 2 times faster, a 3 kg payload, and robot instances with doubled deviations.")


def headline_table(groups: dict) -> str:
    spec = [("hold", "pos_rmse", 1e3, "mrad"), ("free", "pos_rmse", 1e3, "mrad"), ("policy", "pos_rmse", 1e3, "mrad"), ("step", "settle_time", 1.0, "s"),
            ("contact2", "f_peak_20ms", 1.0, "N"), ("contact2", "excess_force_cap", 1.0, "N"), ("task", "f_peak_20ms", 1.0, "N"), ("task", "excess_force_cap", 1.0, "N"),
            ("blocked", "excess_force_cap", 1.0, "N"), ("blocked", "contact_recovery", 1.0, "s"), ("task", "press_deficit", 1.0, "N"), ("contact2", "contact_loss_rate", 1.0, "1/s")]
    rows = []
    for regime, key, scale, unit in spec:
        st = paired_ci(groups, "in_distribution", [PD, RES, PPO_C], regime, key)
        cells = [FAMILY[regime], f"{METRIC_INFO[key][0].split(':')[0].split(' (')[0]} ({unit})", f"{st[PD]['mean'] * scale:.3g}"]
        for m in (RES, PPO_C):
            s = st[m]; lo, hi = s["diff_ci"]; sig = not (np.isnan(lo) or lo <= 0 <= hi)
            cells.append(f"{s['mean'] * scale:.3g}" + ("*" if sig else ""))
        rows.append(cells)
    return table(["Family", "Metric", "PD, shared gains", "PD + learned residual", "PPO, contact families"], rows,
                 caption="Headline numbers behind the first figure: means over the 16 in-distribution cases of each family; a star marks a paired difference against the PD whose 95% bootstrap interval excludes zero. Lower is better throughout.", numeric_cols=(2, 3, 4))


# ------------------------------------------------------------------------------------------ page
def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    groups, d4 = load_pooled(D4)
    ck_groups = load_pooled(CKPT)[0] if CKPT.exists() else None
    hero, traces, grp = fig_hero(groups), fig_traces(), fig_groups(groups)
    blocked = fig_blocked(ck_groups)
    videos = (REPORT / "data" / "videos.html")
    video_html = ""
    if videos.exists():  # keep the clips that matter for the summary: a hold with a pulse, the press-and-slide, the blocked target and the held-out task, at real time
        import re
        blocks = re.findall(r"<figure\b.*?</figure>", videos.read_text(), flags=re.S)
        keep = [b for b in blocks if any(k in b for k in ("fig-video-hold_100x\"", "fig-video-contact2_100x\"", "fig-video-blocked_100x\"", "fig-video-task_100x\""))]
        video_html = "<p>Clips of the same three controllers on representative cases (the gray sphere is the commanded tool position; the plots accumulate as the clip plays).</p>\n" + "\n".join(keep)
    res = groups["in_distribution"]
    g = lambda m, r, k: mean_metric(res[m], r, k)
    n_cases = d4["n_cases"]
    wins = wins_table(d4, [(PD, "PD"), (RES, "residual"), (PPO_C, "PPO contact")], PD,
                      caption="Every (group, family, metric) with a significant paired difference against the PD, for the learned residual and the contact-family PPO arm (16 cases per family and group). Relative change in parentheses; rows at parity are omitted. This is the complete record behind the summary above.")
    blocked_rows = ""
    if ck_groups is not None:
        r = ck_groups["in_distribution"]
        ms = [m for m in (PD, RES, PPO_C, "ppo_cap_gain", "ppo_cap_torque_ext") if m in r]
        blocked_rows = table(["Controller", "Drift from contact point (cm)", "Moving while in contact (-)", "Force above 15 N (N)", "Sustained peak (N)", "Apparent stiffness (N/m)", "Recovery after release (s)"],
                             [[LABEL.get(m, m), f"{mean_metric(r[m], 'blocked', 'blocked_drift') * 100:.1f}", f"{mean_metric(r[m], 'blocked', 'blocked_motion'):.2f}", f"{mean_metric(r[m], 'blocked', 'excess_force_cap'):.1f}",
                               f"{mean_metric(r[m], 'blocked', 'f_peak_20ms'):.0f}", f"{mean_metric(r[m], 'blocked', 'apparent_stiffness'):.0f}", f"{mean_metric(r[m], 'blocked', 'contact_recovery'):.2f}"] for m in ms],
                             caption="Blocked target under 'stop at the surface and wait' (16 in-distribution cases). Drift and motion are new; the force terms and the recovery existed. The PD under the previous definition had looked best on this family because it followed the reference's tangential drift; under the new one the ranking is driven by the force the controller keeps applying.", numeric_cols=(1, 2, 3, 4, 5, 6))

    parts = []
    parts.append(f"""<h2>Summary</h2>
<p>We study whether the low-level feedback controller of a torque-controlled manipulator, the loop that turns position targets into motor torques at 1 kHz, should be learned rather than hand-tuned. On a simulated Franka Panda with randomized link parameters, payloads and sensor noise, a learned controller that adds a bounded neural residual to a tuned PD, with only 100 ms of proprioceptive history as input, tracks position targets more accurately than the tuned PD in every family of references we test and limits contact forces far better when the targets run into a surface: position RMSE {g(RES, 'free', 'pos_rmse') * 1e3:.1f} against {g(PD, 'free', 'pos_rmse') * 1e3:.1f} mrad in free motion, settling time {g(RES, 'step', 'settle_time'):.2f} against {g(PD, 'step', 'settle_time'):.2f} s on steps, sustained peak force {g(RES, 'task', 'f_peak_20ms'):.0f} against {g(PD, 'task', 'f_peak_20ms'):.0f} N on a held-out task sequence. These advantages hold under a shifted workspace, references twice as fast, a 3 kg payload and robot instances with doubled parameter deviations, none of which were used for training, tuning or selection. A reinforcement-learning variant trained on contact pushes the force reductions further at some cost in free-space smoothness.</p>
<p>The learned controllers are not better everywhere. They do not press to a target force when asked, they push as hard as the PD when a target is blocked, and they lose contact more often during a press. We have made the wanted behaviour on a blocked target explicit, stop at the surface and wait without unnecessary force, measure it, and show that no controller does it yet. That is the open problem the next phase addresses.</p>""")

    parts.append("""<h2>1. Motivation</h2>
<p>Almost every robot closes its lowest control loop with a PD law: a stiffness times the position error plus a damping times the velocity error, often with gravity and Coriolis compensation, at 100 Hz to 1 kHz. The gains are tuned once, by hand, for a broad range of tasks. Four observations motivate learning this loop instead.</p>
<ol>
<li>There is no universal gain set. The gains that track a fast free-space motion well are too stiff for a press against a surface; the gains that make contact gentle track poorly; and both change with the payload, with the robot instance and within a task as the contact mode changes.</li>
<li>Policies above the controller treat it as a black box. Much of the jerky or unstable behaviour of well-trained policies in distribution comes from the controller below them: its response depends on payload, dynamics, system identification and contact in ways the policy never sees.</li>
<li>Hand-tuning per scenario does not scale, and switching between gain sets stably is itself a control problem.</li>
<li>A controller learned with a proprioceptive history has a small sim-to-real gap: its inputs are joint positions, velocities and commanded torques, which every torque-controlled arm provides.</li>
</ol>
<p>The research question is therefore: can a learned feedback controller be <em>versatile</em>, better than a tuned PD across free motion, policy-like targets and contact, on robots it has not seen, with contact forces that are safe to execute on hardware?</p>""")

    parts.append("""<h2>2. The plant and the controllers</h2>
<p><strong>Plant.</strong> A 7-joint Franka Panda in MuJoCo, torque-controlled at 1 kHz, with a spherical tool, a table and a wall. Every training window and every evaluation case draws a robot instance (link masses, inertias, joint friction and damping perturbed around the nominal model), a payload at the tool (0 to 3 kg in training, 0 to 2 kg in the test families, 3 kg as a held-out group) and sensor noise on the joint positions. The controller never sees the instance, the payload or any force sensor.</p>
<p><strong>Controllers.</strong> The reference is a PD with gravity compensation whose seven stiffness and damping pairs were tuned by direct search on the protocol's tuning split (stiffness 47 to 395 Nm/rad, damping ratio 0.69). The learned controllers are torque laws of the same signals. We studied three parameterisations: learned gains, PD plus a learned residual torque, and a fully neural torque; and three ways to train them: gradients through the simulator, gradients through a learned dynamics model, and reinforcement learning. The controller featured here is <em>PD plus residual</em>: the tuned PD plus a residual bounded to half the torque limit, from a two-layer network (128 units, 134 k parameters) that sees the current position, velocity and target, and a history of positions, velocities and commanded torques over the last 100 ms at 10 ms spacing and the last 32 ms at 1 ms spacing. It was trained in the simulator by a short-horizon actor-critic method through finite-difference plant gradients (500 iterations of 24 windows of 2 s), with a critic that sees privileged quantities (contact force, payload, a reference preview) during training only. The reinforcement-learning variant fine-tunes the same network with PPO on windows of the contact families.</p>""")

    parts.append("<h2>3. Protocol: what is measured and what is wanted</h2>\n<p>Seven families of references, each with its own metrics and its own definition of the wanted behaviour, scored on 16 test cases per family under paired comparisons with bootstrap confidence intervals. Composites exist for tuning and selection only; every claim below rests on the primary metrics of a family.</p>\n" + protocol_table())

    parts.append("<h2>4. First evidence: learned controllers outperform the tuned PD</h2>\n<p>The first figure summarises the comparison on the in-distribution families; the table gives the numbers; the second figure shows that the same advantages hold under every held-out condition; the traces and the clips show what the controllers do.</p>\n" + hero + headline_table(groups) + grp + traces + video_html)

    parts.append("<h2>5. Where they lose</h2>\n<p>The same tables show three consistent losses. (1) <em>Pressing to a target.</em> When the task family asks for a 1 cm press, the learned controllers press lightly and intermittently: the press deficit is several times the PD's. (2) <em>Blocked targets.</em> When the reference passes through a surface, every controller, learned or not, keeps pushing with the force its stiffness implies; the residual pushes as hard as the PD. (3) <em>Keeping contact.</em> During a press the learned controllers lose contact for 5 to 10 ms several times a second: a learned flinch at the reward's force limit, where the PD base spring pushes the tool back in. The reinforcement-learning variant trades more of (1) and (3) for lower forces. These are the behaviours that make a controller <em>compliant</em> rather than merely accurate, and they are the subject of the current deep dive.</p>\n" + wins)

    parts.append("<h2>6. The blocked target, redefined</h2>\n<p>What should a controller do when its position target is unreachable? The protocol had rewarded following the part of the reference that remains feasible, i.e. sliding along the surface; we now define the wanted response as <em>stop at the surface and wait</em>: hold the point where the tool first touched, apply no more force than the 15 N cap, and resume tracking when the reference returns. The cap itself follows what pick-and-place, peg insertion and light connector tasks need (connector datasheets list 20 to 30 N mating forces; third-party Franka software defaults its collision threshold to 30 N). Two metrics make the definition measurable, the tangential drift from the first-contact point and the fraction of time in motion while blocked, next to the force above the cap, the sustained peak, the apparent stiffness (force per metre of commanded depth) and the recovery after release.</p>\n" + blocked_rows + blocked)

    audit_txt = ""
    if AUDIT.exists():
        a = json.loads(AUDIT.read_text())
        b = a["regimes"]["blocked"]["controllers"]
        audit_txt = f" On blocked windows the privileged exemplar scores {b['pd_soft_inside']['rewards']['cap']['total']:+.2f} reward units per step under the new reward, the PD {b['pd']['rewards']['cap']['total']:+.2f} and the learned residual {b['residual']['rewards']['cap']['total']:+.2f}."
    parts.append(f"""<h2>7. What we learned about training for contact</h2>
<p>Three findings shaped the current approach. First, the reward must name the behaviour: the original reward penalised force above a limit but never asked for a maintained press, and evaluated on saved episodes it preferred intermittent light contact to a steady press. The current reward adds, while the reference is inside a surface, a kernel on the normal force around the force the commanded depth implies (capped at 15 N) and switches off the joint kernel that would ask for the infeasible depth; audited on identical windows against a privileged exemplar (a PD whose gains drop to 0.3 only while the reference is inside), it ranks the exemplar first on both surface families.{audit_txt} Second, the impact transient must not dominate: a blind position controller cannot avoid the impact at the reference speed, and a one-step force penalty of an impact is a hundred times the tracking reward, so gradients through it teach softening everywhere; the force terms now act on a 20 ms filtered force, and the actor-critic route no longer backpropagates through the contact force at all. Third, the data must contain the behaviour: a sign error in the blocked-reference generator had left 60% of the blocked training windows without any block, and the wall references had the hand reach the wall before the tool; both are fixed, with regression tests, and every table in this summary is computed on the corrected protocol.</p>
<p>Two further observations guide the next phase. A linear probe of the featured controller's internal features decodes 'tool in contact' with only 0.76 accuracy and a 0.7 s median delay, so contact is not something the network currently senses promptly; giving it the robot model's external-torque estimate (the signal a Panda exposes on hardware) produced the first fine-tuned arms to beat their warm start on validation, before training was paused for this checkpoint. And a residual parameterised as bounded multipliers of the PD gains, so that yielding is a small action, was never explored by PPO at the exploration width used, which is a property of the optimiser setting, not of the parameterisation.</p>""")

    parts.append("""<h2>8. Next</h2>
<ul>
<li>Train under the stop-and-wait definition: the tangential kernel around the first-contact point in blocked windows, with the external-torque input, three seeds, both training routes; then the model study (feature encoding, output smoothing).</li>
<li>Report compliance as what it is: apparent stiffness in contact, drift and motion while blocked, force above the cap, and the recovery after release, next to the tracking metrics.</li>
<li>Keep the task family held out as the end-to-end test; the multi-task recipe follows once one family is solved.</li>
</ul>""")
    (REPORT / "report.html").write_text("\n\n".join(parts))
    print("wrote", REPORT / "report.html")


if __name__ == "__main__":
    main()
