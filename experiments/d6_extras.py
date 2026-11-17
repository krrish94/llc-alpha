"""D5 (contact deep dive) report extras: the reward-audit table and the training curves of the
actor-critic and PPO arms (validation under the cap on the 32 fixed cases). HTML snippets used by
d6_narrative."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, method_color  # noqa: E402
from report_api import figure, table  # noqa: E402

REPORT = Path("research/reports/11-dd-contact")
LABELS = {
    "pd_residual_cap_torque": "Actor-critic, torque residual, cap reward",
    "pd_gain_residual_cap_gain": "Actor-critic, gain residual, cap reward",
    "pd_residual_v2_torque": "Actor-critic, torque residual, D3 reward (control)",
    "pd_gain_residual_v2_gain": "Actor-critic, gain residual, D3 reward (control)",
    "ppo_cap_torque": "PPO, torque residual, cap reward",
    "ppo_cap_gain": "PPO, gain residual, cap reward",
    "ppo_v2_torque": "PPO, torque residual, D3 reward (control)",
    "ppo_v2_gain": "PPO, gain residual, D3 reward (control)",
    "pd_residual_cap_torque_ext": "Actor-critic, torque residual, cap reward, external-torque input",
    "pd_gain_residual_cap_gain_ext": "Actor-critic, gain residual, cap reward, external-torque input",
    "ppo_cap_torque_ext": "PPO, torque residual, cap reward, external-torque input",
    "ppo_cap_gain_ext": "PPO, gain residual, cap reward, external-torque input",
    "pd_residual_cap_torque_fourier": "Actor-critic, torque residual, cap reward, Fourier features",
    "pd_gain_residual_cap_gain_fourier": "Actor-critic, gain residual, cap reward, Fourier features",
    "pd_residual_cap_torque_filt": "Actor-critic, torque residual, cap reward, 20 ms output filter",
    "pd_gain_residual_cap_gain_filt": "Actor-critic, gain residual, cap reward, 20 ms output filter",
    "ppo_cap_torque_filt": "PPO, torque residual, cap reward, 20 ms output filter",
    "ppo_cap_gain_filt": "PPO, gain residual, cap reward, 20 ms output filter",
    "ppo_cap_torque_fourier": "PPO, torque residual, cap reward, Fourier features",
    "ppo_cap_gain_fourier": "PPO, gain residual, cap reward, Fourier features",
    "ppo_cap_torque_explore": "PPO, torque residual, cap reward, wider exploration (std 0.15, actor lr 3e-4)",
    "ppo_cap_gain_explore": "PPO, gain residual, cap reward, wider exploration (std 0.15, actor lr 3e-4)",
}
AUDIT_LABELS = {"pd": "PD, shared gains", "pd_soft": "PD, gains x0.3 everywhere", "pd_soft_inside": "PD, gains x0.3 only while the reference is inside (privileged exemplar)",
                "residual": "PD + residual, actor-critic (D3 warm start)", "ppo_contact": "PPO, contact families (D4, v2)"}


def method_key(run_dir: Path) -> str:
    """Pooled key of a run as experiments/d_evaluate.py and d_report.py form it."""
    meta = json.loads((run_dir / "meta.json").read_text())
    if "method_key" in meta:
        return meta["method_key"]
    return meta["policy_config"]["kind"] + (f"_{meta['args']['variant']}" if meta["args"].get("variant") else "")


def audit_table(path: Path = Path("runs/d6/reward_audit.json")) -> str:
    if not path.exists():
        return ""
    d = json.loads(path.read_text())
    rows = []
    for regime, label in (("contact2", "Contact (press and slide)"), ("blocked", "Blocked target")):
        ctr = d["regimes"][regime]["controllers"]
        for name in ("pd", "pd_soft", "pd_soft_inside", "residual", "ppo_contact"):
            if name not in ctr:
                continue
            e, st = ctr[name], ctr[name]["stats"]
            rows.append([label, AUDIT_LABELS.get(name, name), f"{e['rewards']['d3']['total']:.3f}", f"{e['rewards']['cap']['total']:.3f}",
                         f"{st['mean force inside (N)']:.1f}", f"{st['contact fraction inside']:.2f}", f"{st['losses per s inside']:.1f}", f"{st['peak force (N)']:.0f}"])
    return table(["Family", "Controller", "D3 reward (per step)", "Cap reward (per step)", "Mean force inside (N)", "Contact fraction inside", "Losses per s inside", "Peak force (N)"], rows,
                 caption="Reward audit on 12 training windows per family (half starting 0.1 to 0.2 s before the reference crosses the surface), identical for every controller: mean step reward under the D3 reward and under the cap reward with the contact-maintenance terms, and what each controller does while the reference is inside the surface. Source: runs/d6/reward_audit.json.",
                 numeric_cols=(2, 3, 4, 5, 6, 7))


def training_curves(runs: list[str]) -> str:
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    seen = set()
    for run in runs:
        rd = Path(run)
        log = rd / "log.jsonl"
        if not log.exists():
            continue
        recs = [json.loads(l) for l in open(log)]
        vals = [r for r in recs if r.get("val")]
        if not vals:
            continue
        key = method_key(rd)
        is_ppo = key.startswith("ppo")
        total = 200 if is_ppo else max(1, max(r.get("iter", 0) for r in recs) + 1)
        x = [((r["update"] + 1) / total if is_ppo else (r["iter"] + 1) / total) for r in vals]
        label = LABELS.get(key, key) if key not in seen else None
        seen.add(key)
        col, ls = method_color(key), ("--" if is_ppo else "-")
        axes[0].plot(x, [r["val"]["mean"] for r in vals], ls, marker="o", ms=2.5, color=col, lw=0.8, label=label)
        axes[1].plot(x, [r["val"]["excess_force_cap"] for r in vals], ls, marker="o", ms=2.5, color=col, lw=0.8)
        axes[2].plot(x, [r["val"].get("press_force", np.nan) for r in vals], ls, marker="o", ms=2.5, color=col, lw=0.8)
        axes[3].plot(x, [r["val"].get("contact_loss_rate", np.nan) for r in vals], ls, marker="o", ms=2.5, color=col, lw=0.8)
    ref = Path("runs/d6/val_pd_shared.json")
    if ref.exists():
        pd = json.loads(ref.read_text())
        for ax, k in zip(axes, ("mean", "excess_force_cap", "press_force", "contact_loss_rate")):
            ax.axhline(pd[k], color="#8c8c8c", lw=0.8, ls=(0, (4, 2)), label="PD, shared gains (v2), same cases" if k == "mean" else None)
    axes[0].set_ylabel("validation composite (cap)"); axes[1].set_ylabel("validation excess force above 15 N (N)")
    axes[2].set_ylabel("mean press-phase force, surface families (N)"); axes[3].set_ylabel("contact losses per s of press phase")
    for ax in axes:
        ax.set_xlabel("fraction of the run (actor-critic solid, PPO dashed)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT / "figures" / "training.svg", bbox_inches="tight")
    plt.close(fig)
    return figure("figures/training.svg", "Training under the cap: validation on 32 fixed protocol v2 cases (two per free-space family, eight per surface family) every 50 actor-critic iterations or 25 PPO updates; the first point is the warm start. Composite under the 15 N cap, excess force above the cap, mean force in press phases and contact losses per second. Gray dashed: the protocol v2 PD on the same cases.", "fig-training", wide=True)


if __name__ == "__main__":
    print(audit_table())
    print(training_curves(sorted(str(p) for p in Path("runs/d6").glob("*-s*") if p.is_dir())))
