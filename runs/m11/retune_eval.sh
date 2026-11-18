#!/bin/zsh
# Milestone 11, Phase 1 (continued, 2026-11-17): the PD re-tuned per profile (phase1_tune.sh) evaluated through its profile, the
# re-tuned gains being the servo's nominal gains; then the velocity-filter sweep (ADR 0011 amendment 3): the classical set, the
# oracle and the P1 host-rate student through P1 with a first-order filter on the velocity feedback at 2, 5 and 10 ms.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
for prof in p1 p2 p3; do
  uv run --no-sync python experiments/d_evaluate.py --gains runs/m11/tune_$prof/gains_shared.json --extra-gains --workers 6 --profile $prof \
    --groups in_distribution tasks torque_tracking surface_500 --out runs/m11/metrics_tune_$prof.json > runs/m11/metrics_tune_$prof.log 2>&1
done
for vf in 0.002 0.005 0.01; do
  uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains pd_shared_cap=runs/m10/gains_shared.json --feedforward --oracle-on pd --workers 6 \
    --profile p1 --vel-filter $vf --runs runs/m11/student_p1-s0 --groups in_distribution torque_tracking --out runs/m11/velfilter_p1_$vf.json > runs/m11/velfilter_p1_$vf.log 2>&1
done
echo "re-tune evaluations and the velocity-filter sweep done at $(date +%H:%M)"
