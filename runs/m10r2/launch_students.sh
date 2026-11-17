#!/bin/zsh
# Milestone 10, round 2: distilled students from the round-2 bases (oracle teacher around the frozen base, press floor 7 N = the
# protocol's floor, item C), twelve DAgger iterations with the teacher driving only in the first two, free-space labels weighted 3x,
# three seeds per base kind. The residual student is the gain-residual form with the external-torque estimate (as in Milestone 10);
# the torque student keeps the direct torque output and adds the external-torque estimate.
# Usage: ./runs/m10r2/launch_students.sh [pd_residual|neural_torque|both]   (seed s of a student starts from base seed s)
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
which=${1:-both}
for kind in pd_residual neural_torque; do
  [[ $which == both || $which == $kind ]] || continue
  if [[ $kind == pd_residual ]]; then tag=distill_ff; extra="--gain-residual"; variant=v3_r2_gain_ext; else tag=distill_torque; extra=""; variant=v3_r2_torque; fi
  for s in 0 1 2; do
    nohup uv run --no-sync python experiments/d7_distill.py --warmstart runs/m10r2/${kind}_ff-s$s/best.pt --gains runs/m9/gains_shared.json ${=extra} --ext-torque \
      --oracle-f_min 7 --iters 12 --betas 1,0 --free-weight 3 --cases 48 --workers 4 --threads 2 \
      --seed $s --variant $variant --out runs/m10r2/$tag-s$s > runs/m10r2/$tag-s$s.log 2>&1 &
  done
done
echo "students ($which) launched at $(date +%H:%M)"
