#!/bin/zsh
# Milestone 9: transfer fine-tuning: the D6 students (trained on the sphere plant) distilled further on the gripper plant, each from its own
# weights with the oracle around itself as the teacher (frozen copy); six DAgger iterations, the teacher driving only in the first one, three seeds.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
for s in 0 1 2; do
  nohup uv run --no-sync python experiments/d7_distill.py --warmstart runs/d7/distill_gain_ext-s$s/best.pt --gains runs/m9/gains_shared.json --gain-residual --ext-torque \
    --iters 6 --betas 1,0,0 --cases 48 --workers 3 --threads 2 --seed $s --variant v3_transfer --out runs/m9/distill_transfer-s$s > runs/m9/distill_transfer-s$s.log 2>&1 &
done
echo "transfer students launched at $(date +%H:%M)"
