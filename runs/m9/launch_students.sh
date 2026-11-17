#!/bin/zsh
# Milestone 9: distilled students on plant v3 from the v3 warm start (oracle teacher around it), with the free-space fix: twelve DAgger iterations,
# the teacher driving only in the first two, three seeds. Usage: ./runs/m9/launch_students.sh <warmstart best.pt>
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
WS=${1:-runs/m9/pd_residual_ac-s0/best.pt}
for s in 0 1 2; do
  nohup uv run --no-sync python experiments/d7_distill.py --warmstart $WS --gains runs/m9/gains_shared.json --gain-residual --ext-torque --iters 12 --betas 1,0 --free-weight 3 --cases 48 --workers 4 --threads 2 \
    --seed $s --variant v3_gain_ext --out runs/m9/distill_gain_ext-s$s > runs/m9/distill_gain_ext-s$s.log 2>&1 &
done
echo "students launched at $(date +%H:%M)"
