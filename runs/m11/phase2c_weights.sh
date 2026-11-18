#!/bin/zsh
# Milestone 11, Phase 2c (the advisor's decision of 2026-11-17: contact-weighted labels). The P1 host-rate student's recipe with the
# label weights and the teacher's share varied, one seed each, read against runs/m11/student_p1-s0 (free weight 3, contact weight 1,
# the teacher driving the first iteration only): the contact tilt, the neutral weights, and the teacher's states kept in the
# aggregate for three more iterations.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
common=(--profile p1 --gain-residual --ext-torque --free-teacher ideal --teacher-base pd --gains runs/m9/gains_shared.json --oracle-f_min 7
        --iters 12 --cases 48 --workers 4 --threads 2 --layers 4 --hidden 256 --seed 0)
uv run --no-sync python experiments/d7_distill.py $common --betas 1,0 --free-weight 1 --contact-weight 3 --variant host_p1_w1c3 --out runs/m11/student_p1_w1c3-s0 > runs/m11/student_p1_w1c3-s0.log 2>&1
uv run --no-sync python experiments/d7_distill.py $common --betas 1,0 --free-weight 1 --contact-weight 1 --variant host_p1_w1c1 --out runs/m11/student_p1_w1c1-s0 > runs/m11/student_p1_w1c1-s0.log 2>&1
uv run --no-sync python experiments/d7_distill.py $common --betas 1,0.5,0.5,0.5,0 --free-weight 3 --contact-weight 1 --variant host_p1_beta --out runs/m11/student_p1_beta-s0 > runs/m11/student_p1_beta-s0.log 2>&1
echo "phase 2c sweep done at $(date +%H:%M)"
