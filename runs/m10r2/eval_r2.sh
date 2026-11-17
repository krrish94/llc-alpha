#!/bin/zsh
# Milestone 10, round 2 benchmark on protocol v3: the classical set (two tuned PDs and their feedforward variants), the oracle ceiling, the
# Milestone 10 student (runs/m9/distill_gain_ext-s*), the round-2 bases (residual with feedforward, torque network) and their students;
# all groups, tasks, the 5 ms latency group; episodes saved. Extra arguments go to d_evaluate (for example --workers 10).
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains "pd_shared_cap=runs/m10/gains_shared.json" --feedforward --oracle-on pd --latency 0.005 \
  --runs runs/m9/distill_gain_ext-s0 runs/m9/distill_gain_ext-s1 runs/m9/distill_gain_ext-s2 \
         runs/m10r2/pd_residual_ff-s0 runs/m10r2/pd_residual_ff-s1 runs/m10r2/pd_residual_ff-s2 runs/m10r2/neural_torque_ff-s0 runs/m10r2/neural_torque_ff-s1 runs/m10r2/neural_torque_ff-s2 \
         runs/m10r2/distill_ff-s0 runs/m10r2/distill_ff-s1 runs/m10r2/distill_ff-s2 runs/m10r2/distill_torque-s0 runs/m10r2/distill_torque-s1 runs/m10r2/distill_torque-s2 \
         runs/m10r2/distill_two-s0 runs/m10r2/distill_two-s1 runs/m10r2/distill_two-s2 \
  --out runs/m10r2/metrics_test.json --save-episodes runs/m10r2/episodes_test "$@"
