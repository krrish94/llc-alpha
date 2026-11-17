#!/bin/zsh
# Milestone 10, round 2, item D: two-teacher students from the feedforward residual bases. Free-space families labelled with the
# TAM-style ideal-robot torque of the student's own drive (hold pulses included as external loads), contact families with the
# oracle (7 N floor); otherwise the round-2 student recipe (gain residual, external-torque input, twelve iterations, free weight 3).
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
for s in 0 1 2; do
  nohup uv run --no-sync python experiments/d7_distill.py --warmstart runs/m10r2/pd_residual_ff-s$s/best.pt --gains runs/m9/gains_shared.json --gain-residual --ext-torque \
    --free-teacher ideal --oracle-f_min 7 --iters 12 --betas 1,0 --free-weight 3 --cases 48 --workers 4 --threads 2 \
    --seed $s --variant v3_r2_two_teachers --out runs/m10r2/distill_two-s$s > runs/m10r2/distill_two-s$s.log 2>&1 &
done
echo "two-teacher students launched at $(date +%H:%M)"
