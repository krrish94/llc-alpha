#!/bin/zsh
# Milestone 11, Phase 2b (ADR 0011, decision 5): host-rate students per profile on the Panda plant v3, distilled from scratch by the
# two-teacher recipe (ideal-robot labels in free space, the oracle around the feedforward PD in contact, 7 N floor) through the
# profile's servo loop with the model-based interface; the student decides at every host frame, no compute latency, 4 layers of 256.
# Usage: ./runs/m11/phase2_students.sh [seed]   (profiles in sequence)
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
s=${1:-0}
for prof in p1 p2 p3; do
  uv run --no-sync python experiments/d7_distill.py --profile $prof --gain-residual --ext-torque --free-teacher ideal --teacher-base pd --gains runs/m9/gains_shared.json \
    --oracle-f_min 7 --iters 12 --betas 1,0 --free-weight 3 --cases 48 --workers 4 --threads 2 --layers 4 --hidden 256 \
    --seed $s --variant host_$prof --out runs/m11/student_$prof-s$s > runs/m11/student_$prof-s$s.log 2>&1
done
echo "phase 2b (seed $s) done at $(date +%H:%M)"
