#!/bin/zsh
# Milestone 9: the retrained controllers on protocol v3 (native v3 warm start and students, transfer students, the v2 student unchanged) against the baselines; all groups, episodes saved.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains "pd_shared_v2gains=runs/d1/gains_shared.json" --feedforward --oracle-on pd \
  --runs runs/d7/distill_gain_ext-s0 runs/d7/distill_gain_ext-s1 runs/d7/distill_gain_ext-s2 \
         runs/m9/distill_transfer-s0 runs/m9/distill_transfer-s1 runs/m9/distill_transfer-s2 \
         runs/m9/pd_residual_ac-s0 runs/m9/distill_gain_ext-s0 runs/m9/distill_gain_ext-s1 runs/m9/distill_gain_ext-s2 \
  --out runs/m9/metrics_students.json --save-episodes runs/m9/episodes_students "$@"
