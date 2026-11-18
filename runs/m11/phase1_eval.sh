#!/bin/zsh
# Milestone 11, Phase 1 (ADR 0011): the interface ablation on plant v3. The classical set (two tuned PDs, their feedforward variants),
# the oracle ceiling and the torque-tracking program run through every profile with the model-based interface; then the feedforward PD
# over a sweep of host periods and bus delays (hypothesis P1); then the PD re-tuned per profile by CMA-ES (same budget as Milestone 9,
# cap composite, feedforward form) runs in phase1_tune.sh.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
G="--gains runs/m9/gains_shared.json --extra-gains pd_shared_cap=runs/m10/gains_shared.json --feedforward --oracle-on pd --workers 12"
for prof in p0 p1 p2 p3; do
  uv run --no-sync python experiments/d_evaluate.py ${=G} --profile $prof --groups in_distribution tasks torque_tracking surface_500 --out runs/m11/metrics_$prof.json
done
# the sweep: a position interface with RAM gains (P2-like) over periods at 2 ms delay, and over delays at a 10 ms period
for per in 0.002 0.004 0.02 0.04; do
  uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains --feedforward --workers 12 --profile p2 --period $per --groups in_distribution torque_tracking --out runs/m11/sweep_period_${per}.json
done
for del in 0.0 0.005 0.01; do
  uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains --feedforward --workers 12 --profile p2 --delay $del --groups in_distribution torque_tracking --out runs/m11/sweep_delay_${del}.json
done
echo "phase 1 evaluations done at $(date +%H:%M)"
