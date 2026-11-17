"""Build the Milestone 5 report (research/reports/05-ppo) from runs/m5/metrics_test.json (PPO runs
evaluated with experiments/m2_evaluate.py) and the PPO/BPTT training logs: performance on the
same test cases, training cost in environment steps and wall time, and sample-efficiency curves."""
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
from report_api import compare_figure, figure, fmt, table  # noqa: E402

from experiments.m2_report import BASELINES, KIND_LABELS, REGIME_LABELS, build_compare, primary_table  # noqa: E402
from llc.eval.cases import REGIMES  # noqa: E402

REPORT = Path("research/reports/05-ppo")


def cost_table(runs: dict[str, Path]) -> str:
    rows = []
    for name, path in runs.items():
        recs = [json.loads(l) for l in open(path / "log.jsonl") if l.strip()]
        meta = json.load(open(path / "meta.json"))
        last = recs[-1]
        if "env_steps" in last:
            rows.append([name, "PPO (plant)", str(last["env_steps"]), str(last["physics_steps"]), f"{last['elapsed_s']/60:.0f}", str(len([r for r in recs if 'val' in r]))])
        else:
            iters = [r for r in recs if "iter" in r]
            batch, horizon = meta["args"]["batch"], meta["rollout"]["horizon_s"]
            model_steps = len(iters) * batch * int(round(horizon / meta["rollout"]["dt"]))
            rows.append([name, "BPTT (known model)", "0 (model only)", str(model_steps) + " model steps", f"{iters[-1]['elapsed_s']/60:.0f}", str(len([r for r in iters if 'val' in r]))])
    return table(["Run", "Route", "Plant environment steps", "Physics / model steps", "Wall time (min)", "Validation evaluations"], rows,
                 caption="Training cost. PPO steps are network periods (10 ms) in the plant; BPTT steps are 2.5 ms model steps. Validation evaluations use the same 12 cases in MuJoCo for both routes.", numeric_cols=(2, 3, 4, 5))


def efficiency_figure(runs: dict[str, Path]) -> str:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for name, path in runs.items():
        recs = [json.loads(l) for l in open(path / "log.jsonl") if l.strip()]
        vals = [(r.get("env_steps", None), r["elapsed_s"], r["val"]["mean"]) for r in recs if "val" in r]
        if not vals:
            continue
        kind = name.split("-")[0].replace("ppo_", "").replace("_long", "").replace("_lownoise", "")
        ls = "--" if "ppo" in name else "-"
        axes[1].plot([v[1] / 60 for v in vals], [v[2] for v in vals], ls, marker="o", ms=3, lw=0.9, color=method_color(kind), label=name)
        if vals[0][0] is not None:
            axes[0].plot([v[0] for v in vals], [v[2] for v in vals], ls, marker="o", ms=3, lw=0.9, color=method_color(kind), label=name)
    axes[0].set_xlabel("plant environment steps (PPO only)"); axes[0].set_ylabel("validation composite (MuJoCo)"); axes[0].set_yscale("log")
    axes[1].set_xlabel("wall time (min)"); axes[1].set_yscale("log")
    axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT / "figures" / "efficiency.svg")
    plt.close(fig)
    return figure("figures/efficiency.svg", "Validation composite in the plant against environment steps (PPO) and wall time (both routes, dashed = PPO). Lower is better.", "fig-efficiency", wide=True)


def main(narrative: dict[str, str], metrics_path: str, runs: dict[str, str]) -> None:
    data = json.load(open(metrics_path))
    groups: dict[str, list[str]] = {}
    for name, info in data["learned"].items():
        route = "PPO" if "m5" in info["path"] else "BPTT"
        groups.setdefault(f"{info['policy_kind']} ({route})", []).append(name)
    results = {k: v for k, v in data["results"].items() if k in dict(BASELINES)}
    for key, names in groups.items():
        n = len(data["results"][names[0]])
        results[key] = [{**data["results"][names[0]][i], **{m: float(np.nanmean([data["results"][nm][i][m] for nm in names])) for m in data["results"][names[0]][i] if isinstance(data["results"][names[0]][i][m], float)}} for i in range(n)]
    methods = BASELINES + [(k, k.replace("learned_gains", KIND_LABELS["learned_gains"]).replace("pd_residual", KIND_LABELS["pd_residual"]).replace("neural_torque", KIND_LABELS["neural_torque"])) for k in groups]
    (REPORT / "data").mkdir(parents=True, exist_ok=True)
    cmp = build_compare(results, methods)
    json.dump(cmp, open(REPORT / "data" / "compare.json", "w"))
    rows = []
    for key, names in groups.items():
        for nm in names:
            res = data["results"][nm]
            rows.append([key, str(data["learned"][nm]["seed"])] + [f"{np.mean([m['composite'] for m in res if m['regime'] == r]):.2f}" for r in REGIMES] + [str(int(sum(m["diverged"] for m in res)))])
    for key, label in BASELINES:
        res = data["results"][key]
        rows.append([label, "—"] + [f"{np.mean([m['composite'] for m in res if m['regime'] == r]):.2f}" for r in REGIMES] + [str(int(sum(m["diverged"] for m in res)))])
    parts = [f"<h2>Question</h2>\n{narrative['question']}", f"<h2>Setup</h2>\n{narrative['setup']}",
             "<h2>Training cost</h2>\n" + narrative["cost"] + cost_table({k: Path(v) for k, v in runs.items()}) + efficiency_figure({k: Path(v) for k, v in runs.items()}),
             "<h2>Results in the plant</h2>\n" + narrative["results_intro"] +
             table(["Controller (route)", "Seed"] + [REGIME_LABELS[r] for r in REGIMES] + ["Diverged"], rows, caption="Composite per regime on the test split (16 cases per regime).", numeric_cols=tuple(range(2, 9))) +
             primary_table(cmp, methods, "Primary metrics (mean over cases; paired difference against PD with shared gains with 95% bootstrap interval).") +
             compare_figure("data/compare.json", cmp, "Per-case metric values on the test split for PPO and BPTT controllers and the baselines.", "fig-compare", default_regime="step", default_metric="pos_rmse"),
             f"<h2>Findings</h2>\n{narrative['findings']}", f"<h2>Caveats and failures</h2>\n{narrative['caveats']}", f"<h2>Next steps</h2>\n{narrative['next']}"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report.html").write_text("\n\n".join(parts))
    print("wrote", REPORT / "report.html")


if __name__ == "__main__":
    from experiments.m5_narrative import METRICS_PATH, NARRATIVE, RUNS

    main(NARRATIVE, METRICS_PATH, RUNS)
