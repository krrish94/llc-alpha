#!/bin/zsh
# Milestone 12 evaluation (ADR 0012, decision 6; protocol v5): 32 cases per family on every group (the v3 and v4 groups, the tasks
# with the five new families, the held-out base stiffness x0.5 and x2, the 3 ms actuator lag, the compute latencies 5 and 10 ms),
# the global classical set (the tracking-tuned and cap-tuned PDs and their feedforward variants), the per-family PD, the oracle
# around the tracking-tuned feedforward PD, and the students (4 layers, 10 layers, 4 layers at the Milestone 10 budget). Waits for
# the students and the per-family tunes.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
until [ -f runs/m12/student_l4_small-s0/final.pt ] && grep -q "done at" runs/m12/tune_families_v5.log 2>/dev/null; do sleep 120; done
uv run --no-sync python experiments/d_evaluate.py --n-cases 32 --gains runs/m9/gains_shared.json --extra-gains pd_shared_cap=runs/m10/gains_shared.json --feedforward --oracle-on pd \
  --family-gains-dir runs/m12 --latency 0.005 0.01 --workers 10 --runs runs/m12/student_l4-s0 runs/m12/student_l10-s0 runs/m12/student_l4_small-s0 \
  --out runs/m12/metrics_test.json --save-episodes runs/m12/episodes_test > runs/m12/eval.log 2>&1
echo "evaluation done at $(date +%H:%M)"
