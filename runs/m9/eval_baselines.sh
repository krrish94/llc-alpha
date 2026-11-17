#!/bin/zsh
# Milestone 9: protocol v3 baselines on plant v3 (shared PD re-tuned, feedforward PD, oracle around the PD), all groups, episodes saved.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains "pd_shared_v2gains=runs/d1/gains_shared.json" --feedforward --oracle-on pd \
  --out runs/m9/metrics_baselines.json --save-episodes runs/m9/episodes_baselines "$@"
