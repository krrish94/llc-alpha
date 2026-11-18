#!/bin/zsh
# Milestone 11 (ADR 0011 amendment 3, item 3): where the torque rate through a profile comes from. The feedforward PD through P1
# with the feedback quantisation removed (exact position and velocity), with the hold and the bus delay removed (a frame every
# millisecond, no delay), and with both removed; read against runs/m11/metrics_p1.json (P1 as modelled) and metrics_p0.json.
# Waits for the Phase 2c evaluation chain.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
until [ -f runs/m11/metrics_retuned_p3.json ]; do sleep 60; done
G="--gains runs/m9/gains_shared.json --extra-gains --feedforward --workers 6 --profile p1 --groups in_distribution"
uv run --no-sync python experiments/d_evaluate.py ${=G} --profile-override pos_res=0 vel_res=0 --out runs/m11/rate_exact_feedback.json > runs/m11/rate_exact_feedback.log 2>&1
uv run --no-sync python experiments/d_evaluate.py ${=G} --period 0.001 --delay 0 --out runs/m11/rate_no_hold.json > runs/m11/rate_no_hold.log 2>&1
uv run --no-sync python experiments/d_evaluate.py ${=G} --period 0.001 --delay 0 --profile-override pos_res=0 vel_res=0 --out runs/m11/rate_neither.json > runs/m11/rate_neither.log 2>&1
uv run --no-sync python experiments/d_evaluate.py ${=G} --profile-override gain_bits=0 tau_bits=0 goal_res=0 vel_cmd_res=0 --out runs/m11/rate_exact_command.json > runs/m11/rate_exact_command.log 2>&1
echo "torque-rate decomposition done at $(date +%H:%M)"
