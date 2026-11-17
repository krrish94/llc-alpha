#!/bin/zsh
# Milestone 10 benchmark on protocol v3: classical set (two tuned PDs and their feedforward variants), the oracle ceiling, and the learned methods
# (actor-critic residual, distilled students, TAM-style ideal residual, the transferred v2 students), all groups, tasks, the 5 ms latency group; episodes saved.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains "pd_shared_cap=runs/m10/gains_shared.json" --feedforward --oracle-on pd --latency 0.005 \
  --runs runs/m9/pd_residual_ac-s0 runs/m9/pd_residual_ac-s1 runs/m9/pd_residual_ac-s2 runs/m9/distill_gain_ext-s0 runs/m9/distill_gain_ext-s1 runs/m9/distill_gain_ext-s2 \
         runs/m10/ideal-s0@ckpt_000.pt runs/m10/ideal-s1@ckpt_000.pt runs/m10/ideal-s2@ckpt_000.pt runs/d7/distill_gain_ext-s0 runs/d7/distill_gain_ext-s1 runs/d7/distill_gain_ext-s2 \
  --out runs/m10/metrics_test.json --save-episodes runs/m10/episodes_test "$@"
