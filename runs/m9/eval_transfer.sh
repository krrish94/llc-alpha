#!/bin/zsh
# Milestone 9: do the plant v2 controllers transfer to plant v3? The D3 residual and the D6 student on the v3 protocol (in-distribution families and the tasks).
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains "pd_shared_v2gains=runs/d1/gains_shared.json" --oracle-on pd \
  --runs runs/d4/pd_residual_acmed_s02_lowrate-s0 runs/d7/distill_gain_ext-s0 runs/d7/distill_gain_ext-s1 runs/d7/distill_gain_ext-s2 \
  --groups in_distribution tasks --out runs/m9/metrics_transfer.json "$@"
