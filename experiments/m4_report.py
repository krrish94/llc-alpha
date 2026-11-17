"""Build the Milestone 4 report (research/reports/04-learned-model): model validation (prediction
error per horizon and regime, gradient agreement), and closed-loop results of controllers trained
through the learned model against those trained through the known model (from a metrics file
produced by experiments/m2_evaluate.py with both sets of runs)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("research").resolve()))
from report_api import compare_figure, fmt, table  # noqa: E402

from experiments.m2_report import BASELINES, KIND_LABELS, REGIME_LABELS, build_compare, primary_table, seed_table  # noqa: E402
from llc.eval.cases import REGIMES  # noqa: E402

RUN_DIR = Path("runs/m4")
REPORT = Path("research/reports/04-learned-model")


def validation_tables(models: dict[str, dict]) -> str:
    rows = []
    for name, v in models.items():
        for r in REGIMES:
            if r in v["prediction_rmse_rad"]:
                p = v["prediction_rmse_rad"][r]
                rows.append([name, REGIME_LABELS[r]] + [fmt(p[h]) for h in ("1 step (2.5 ms)", "100 ms", "1 s")])
    t1 = table(["Model", "Regime", "RMSE after 1 step (rad)", "after 100 ms (rad)", "after 1 s (rad)"], rows,
               caption="Open-loop prediction error on held-out validation episodes (recorded torques replayed; mean over 20 starts per episode).", numeric_cols=(2, 3, 4))
    rows = [[name, str(v["gradient_agreement"]["n"]), fmt(v["gradient_agreement"]["cosine_mean"], 3), fmt(v["gradient_agreement"]["cosine_min"], 3), fmt(v["gradient_agreement"]["rel_norm_error_mean"], 3),
             str(v["n_train_transitions"]), fmt(v["log"][-1]["val_qdd_rmse"], 3)] for name, v in models.items()]
    t2 = table(["Model", "Points", "Cosine similarity (mean)", "Cosine (min)", "Relative norm error", "Training transitions", "Validation acceleration RMSE (rad/s^2)"], rows,
               caption="Gradient usefulness: sensitivity of the joint velocity after 100 ms to a held torque perturbation, learned model against the known rigid-body model, at free-space validation points.", numeric_cols=(1, 2, 3, 4, 5, 6))
    return t1 + t2


def main(narrative: dict[str, str]) -> None:
    models = {name: json.load(open(RUN_DIR / f"model_{name}" / "validation.json")) for name in ("blackbox", "residual") if (RUN_DIR / f"model_{name}" / "validation.json").exists()}
    parts = [f"<h2>Question</h2>\n{narrative['question']}", f"<h2>Setup</h2>\n{narrative['setup']}",
             "<h2>Model validation</h2>\n" + narrative["validation"] + validation_tables(models)]
    metrics_path = RUN_DIR / "metrics_test.json"
    if metrics_path.exists():
        data = json.load(open(metrics_path))
        # label learned runs by training route: runs under runs/m4 are trained through the learned model
        groups: dict[str, list[str]] = {}
        for name, info in data["learned"].items():
            route = ("learned model, black-box" if "_bb-" in name else "learned model, residual") if info["path"].startswith("runs/m4") else "known model"
            key = f"{info['policy_kind']} ({route})"
            groups.setdefault(key, []).append(name)
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
        parts.append("<h2>Closed-loop results in the plant</h2>\n" + narrative["closed_loop"] +
                     table(["Controller (training route)", "Seed"] + [REGIME_LABELS[r] for r in REGIMES] + ["Diverged"], rows, caption="Composite per regime on the test split (16 cases per regime) for controllers trained through the learned model and through the known model.", numeric_cols=tuple(range(2, 9))) +
                     primary_table(cmp, methods, "Primary metrics (mean over cases; paired difference against PD with shared gains with 95% bootstrap interval).") +
                     compare_figure("data/compare.json", cmp, "Per-case metric values on the test split for both training routes.", "fig-compare", default_regime="smooth", default_metric="pos_rmse"))
    parts += [f"<h2>Findings</h2>\n{narrative['findings']}", f"<h2>Caveats and failures</h2>\n{narrative['caveats']}", f"<h2>Next steps</h2>\n{narrative['next']}"]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "report.html").write_text("\n\n".join(parts))
    print("wrote", REPORT / "report.html")


if __name__ == "__main__":
    from experiments.m4_narrative import NARRATIVE

    main(NARRATIVE)
