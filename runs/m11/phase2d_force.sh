#!/bin/zsh
# Milestone 11, Phase 2d (ADR 0011 amendment 3, item 2, the third option): the P1 host-rate student's recipe with the force-space
# label term (the force along the surface normal that the torque error implies, per the 15 N cap, at decisions in contact) at
# weight 1 (the 0.3 run was dropped for time on 2026-11-17 evening), one seed; the label weights and the teacher's share as in the recipe.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
until [ -f runs/m11/student_p1_beta-s0/final.pt ]; do sleep 60; done
common=(--profile p1 --gain-residual --ext-torque --free-teacher ideal --teacher-base pd --gains runs/m9/gains_shared.json --oracle-f_min 7
        --iters 12 --cases 48 --workers 4 --threads 2 --layers 4 --hidden 256 --betas 1,0 --free-weight 3 --contact-weight 1 --seed 0)
uv run --no-sync python experiments/d7_distill.py $common --force-weight 1.0 --variant host_p1_f1 --out runs/m11/student_p1_f1-s0 > runs/m11/student_p1_f1-s0.log 2>&1
echo "phase 2d force-term runs done at $(date +%H:%M)"
