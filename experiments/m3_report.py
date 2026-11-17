"""Build the Milestone 3 report (research/reports/03-robustness) from runs/m3/robustness.json:
robustness curves per factor (composite and position RMSE against the factor level, one line per
controller, with 95% bootstrap intervals over cases) and a table of degradation at held-out levels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("research").resolve()))
from plotstyle import apply_style, method_color  # noqa: E402
from report_api import bootstrap_ci, figure, fmt, table  # noqa: E402

RUN_DIR = Path("runs/m3")
REPORT = Path("research/reports/03-robustness")
FACTOR_LABELS = {"payload_mass": "Payload mass (kg)", "payload_lateral": "Payload lateral offset (m, with 1 kg)", "inertia_scale": "Link inertia scale",
                 "friction_loss": "Joint dry friction (Nm)", "actuation_delay": "Actuation delay (s)", "obs_noise": "Position noise std (rad)", "speed_scale": "Reference speed scale"}
TRAIN_RANGE = {"payload_mass": (0.0, 2.0), "payload_lateral": (0.0, 0.06), "inertia_scale": (0.8, 1.3), "friction_loss": (0.0, 1.0), "actuation_delay": (0.0025, 0.0125),
               "obs_noise": (0.0, 0.002), "speed_scale": (0.7, 1.4)}


def method_label(name: str, labels: dict) -> str:
    return labels.get(name, name)


def palette_key(name: str) -> str:
    base = name.split("-s")[0]
    return base


def curves(data: dict, labels: dict, metric: str, ylabel: str) -> str:
    apply_style()
    factors = list(data["factors"])
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4))
    axes = axes.flatten()
    for ax, factor in zip(axes, factors):
        levels = [float(l) for l in data["sweeps"][factor]]
        for name in data["controllers"]:
            means, lo, hi = [], [], []
            for l in data["sweeps"][factor]:
                vals = [m[metric] for m in data["sweeps"][factor][l][name] if np.isfinite(m[metric])]
                means.append(np.mean(vals)); c = bootstrap_ci(vals, f"{factor}/{l}/{name}/{metric}"); lo.append(c[0]); hi.append(c[1])
            ax.plot(levels, means, "o-", ms=3, lw=1, color=method_color(palette_key(name)), label=method_label(name, labels))
            ax.fill_between(levels, lo, hi, color=method_color(palette_key(name)), alpha=0.15, lw=0)
        a, b = TRAIN_RANGE[factor]
        ax.axvspan(a, b, color="#f0f0f0", zorder=0)
        ax.set_xlabel(FACTOR_LABELS[factor], fontsize=8)
        ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    axes[4].set_ylabel(ylabel)
    axes[-1].axis("off")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    path = f"figures/robustness_{metric}.svg"
    fig.savefig(REPORT / path)
    plt.close(fig)
    return figure(path, f"{ylabel} against each perturbation factor (mean over cases of all regimes with 95% bootstrap interval; log scale). The grey band is the randomization range used for DR training; levels outside it are held out.", f"fig-robust-{metric}", wide=True)


def degradation_table(data: dict, labels: dict) -> str:
    rows = []
    for factor in data["factors"]:
        levels = list(data["sweeps"][factor])
        nominal = "1.0" if factor in ("inertia_scale", "speed_scale") else "0.0"
        for name in data["controllers"]:
            base = np.mean([m["composite"] for m in data["sweeps"][factor][nominal][name]])
            worst_level = levels[-1]
            worst = np.mean([m["composite"] for m in data["sweeps"][factor][worst_level][name]])
            div = sum(m["diverged"] for l in levels for m in data["sweeps"][factor][l][name])
            rows.append([FACTOR_LABELS[factor], method_label(name, labels), fmt(base, 3), f"{worst_level}", fmt(worst, 3), f"{worst / base:.2f}x" if base > 0 else "—", str(int(div))])
    return table(["Factor", "Controller", "Composite at nominal", "Extreme level", "Composite at extreme", "Ratio", "Diverged cases (all levels)"], rows,
                 caption="Degradation from the nominal level to the most extreme (held-out) level of each factor; composite averaged over all regimes and cases.", numeric_cols=(2, 3, 4, 5, 6), css_class="wide")


def combination_table(data: dict, labels: dict) -> str:
    rows = []
    for name in data["controllers"]:
        comps = [np.mean([m["composite"] for m in c["results"][name]]) for c in data["combinations"]]
        div = sum(m["diverged"] for c in data["combinations"] for m in c["results"][name])
        ci = bootstrap_ci(comps, f"combos/{name}")
        rows.append([method_label(name, labels), fmt(np.mean(comps), 3), f"[{fmt(ci[0], 3)}, {fmt(ci[1], 3)}]", fmt(max(comps), 3), str(int(div))])
    return table(["Controller", "Mean composite", "95% CI", "Worst combination", "Diverged cases"], rows,
                 caption=f"Random held-out combinations of all factors ({len(data['combinations'])} combinations, one case per regime each).", numeric_cols=(1, 2, 3, 4))


def main(narrative: dict[str, str], labels: dict[str, str]) -> None:
    data = json.load(open(RUN_DIR / "robustness.json"))
    parts = [f"<h2>Question</h2>\n{narrative['question']}", f"<h2>Setup</h2>\n{narrative['setup']}",
             "<h2>Robustness curves</h2>\n" + narrative["curves"] + curves(data, labels, "composite", "composite") + curves(data, labels, "pos_rmse", "position RMSE (rad)"),
             "<h2>Degradation at held-out levels</h2>\n" + degradation_table(data, labels) + combination_table(data, labels),
             f"<h2>Findings</h2>\n{narrative['findings']}", f"<h2>Caveats and failures</h2>\n{narrative['caveats']}", f"<h2>Next steps</h2>\n{narrative['next']}"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report.html").write_text("\n\n".join(parts))
    print("wrote", REPORT / "report.html")


if __name__ == "__main__":
    from experiments.m3_narrative import LABELS, NARRATIVE

    main(NARRATIVE, LABELS)
