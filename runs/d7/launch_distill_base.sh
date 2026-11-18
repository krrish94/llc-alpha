#!/bin/zsh
# D6 stage 1: distil the oracle (around the frozen D3 warm start) into the gain-residual student with external-torque input; three seeds.
cd "$(dirname "$0")/../.."  # the repository root
for s in 0 1 2; do
  PYTHONPATH=. nohup uv run --no-sync python experiments/d7_distill.py --warmstart runs/d4/pd_residual_acmed_s02_lowrate-s0/best.pt \
    --gain-residual --ext-torque --iters 8 --betas 1,0.5,0.25,0 --cases 48 --workers 3 --threads 2 --seed $s --variant gain_ext --out runs/d7/distill_gain_ext-s$s > runs/d7/distill_gain_ext-s$s.log 2>&1 &
done
