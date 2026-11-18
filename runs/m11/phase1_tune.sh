#!/bin/zsh
# Milestone 11, Phase 1: the PD re-tuned per profile by CMA-ES (Milestone 9 budget, cap composite, feedforward form) through the profile's servo loop.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
for prof in p3 p2 p1; do
  uv run --no-sync python experiments/m1_tune_gains.py --target shared --protocol v2 --objective composite_cap --feedforward --profile $prof --n-cases 4 --iters 40 --popsize 12 --out-dir runs/m11/tune_$prof > runs/m11/tune_$prof.log 2>&1
done
echo "phase 1 tunes done at $(date +%H:%M)"
