#!/bin/zsh
# Milestone 11, Phase 2a (ADR 0011, decision 4): the learned torque-command interface per profile, trained through the plant on the
# torque-tracking loss under the feedforward PD (plus torque programs on hold windows), warm-started at the model-based interface. The first p2 run with the rate weight 0.003 traded tracking for smoothness (kept as iface_p2_ratew-s0); the chain runs with 0.0003.
# Three profiles in sequence, one seed each (seeds 1 and 2 of the best profile follow in phase2_students.sh).
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
for prof in p2 p1 p3; do
  uv run --no-sync python experiments/m11_interface.py --profile $prof --gains runs/m9/gains_shared.json --iters 200 --batch 16 --layers 4 --hidden 256 \
    --w-rate 0.0003 --val-every 25 --threads 3 --sim-threads 4 --workers 4 --seed 0 --out runs/m11/iface_$prof-s0 > runs/m11/iface_$prof-s0.log 2>&1
done
echo "phase 2a done at $(date +%H:%M)"
