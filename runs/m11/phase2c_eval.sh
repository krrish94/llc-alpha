#!/bin/zsh
# Milestone 11, Phase 2c: evaluate the label-weight and teacher-share variants of the P1 student (phase2c_weights.sh) through P1 next
# to the classical set, the oracle and the recipe's student, on every group of protocol v4. Waits for the three runs to finish.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
for v in w1c3 w1c1 beta; do
  until [ -f runs/m11/student_p1_$v-s0/final.pt ]; do sleep 60; done
done
uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains pd_shared_cap=runs/m10/gains_shared.json pd_retuned=runs/m11/tune_p1/gains_shared.json --feedforward --oracle-on pd --workers 10 \
  --profile p1 --runs runs/m11/student_p1-s0 runs/m11/student_p1_w1c3-s0 runs/m11/student_p1_w1c1-s0 runs/m11/student_p1_beta-s0 \
  --out runs/m11/metrics_phase2c_p1.json --save-episodes runs/m11/episodes_phase2c_p1 > runs/m11/metrics_phase2c_p1.log 2>&1
# the PD re-tuned through P2 and P3 (its own gains as the servo's spring: through P3 the servo's gains are the fixed ones) on every group,
# merged as pd_retuned into the students' tables so that the criterion's classical set includes it
for prof in p2 p3; do
  uv run --no-sync python experiments/d_evaluate.py --gains runs/m11/tune_$prof/gains_shared.json --extra-gains --workers 10 --profile $prof --out runs/m11/metrics_retuned_$prof.json > runs/m11/metrics_retuned_$prof.log 2>&1
  python3 - "$prof" <<'PY'
import json, sys
prof = sys.argv[1]
main = json.load(open(f"runs/m11/metrics_students_{prof}.json")); new = json.load(open(f"runs/m11/metrics_retuned_{prof}.json"))
for g, res in new["groups"].items():
    if g in main["groups"] and "pd_shared" in res:
        main["groups"][g]["pd_retuned"] = res["pd_shared"]
main["controllers"]["pd_retuned"] = new["controllers"]["pd_shared"]
main["retuned_merge"] = "pd_retuned rows: the PD re-tuned through the profile (runs/m11/tune_%s), evaluated with its own gains as the servo's spring and merged (2026-11-17)" % prof
json.dump(main, open(f"runs/m11/metrics_students_{prof}.json", "w"), indent=1)
print("merged", prof)
PY
done
echo "phase 2c evaluation done at $(date +%H:%M)"
