#!/bin/zsh
# Milestone 11, Phase 2d: evaluate the force-term variants of the P1 student (phase2d_force.sh) through P1 next to the classical set
# (with the re-tuned PD), the oracle and the recipe's student, on every group of protocol v4. Waits for both runs.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
for v in f1; do
  until [ -f runs/m11/student_p1_$v-s0/final.pt ]; do sleep 60; done
done
uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains pd_shared_cap=runs/m10/gains_shared.json pd_retuned=runs/m11/tune_p1/gains_shared.json --feedforward --oracle-on pd --workers 10 \
  --profile p1 --runs runs/m11/student_p1-s0 runs/m11/student_p1_f1-s0 \
  --out runs/m11/metrics_phase2d_p1.json --save-episodes runs/m11/episodes_phase2d_p1 > runs/m11/metrics_phase2d_p1.log 2>&1
echo "phase 2d evaluation done at $(date +%H:%M)"
