#!/bin/zsh
# Milestone 9: the residual warm start on plant v3 by the D3 recipe (actor-critic through the plant, bounded kernels, median clipping), v3 gains.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3 LLC_VAL_WORKERS=4
mkdir -p runs/m9/pd_residual_ac-s0
nohup uv run --no-sync python experiments/d2_train.py --kind pd_residual --gains runs/m9/gains_shared.json --objective ac --ac-horizon 100 --median-clip 3 --pos-sigma 0.02 --w-rate 0.003 \
  --lr 1e-3 --stage-iters 100 100 300 --batch 24 --horizon 2.0 --val-every 50 --val-cases 2 --val-cases-contact 8 --val-objective composite_cap --threads 3 --sim-threads 6 \
  --seed 0 --variant v3 --out runs/m9/pd_residual_ac-s0 > runs/m9/pd_residual_ac-s0/train.log 2>&1 &
echo "warm start launched at $(date +%H:%M)"
