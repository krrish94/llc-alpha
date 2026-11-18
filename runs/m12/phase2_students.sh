#!/bin/zsh
# Milestone 12, phase 2 (ADR 0012, decisions 3 to 5): the two-phase students distilled at scale from their free-space bases by DAgger
# from two teachers (the ideal-robot labels in free space, the oracle around the warm start in contact), on the consolidation
# mixture of the fourteen families, under the v5 training distribution (the base gains and the latency drawn per episode and given
# to the student as inputs, expanded into the warm start with zero columns), twenty iterations of 320 episodes with no cap on the
# label store, the teacher's share 1, then 0.5 for four iterations, 0.25 for four, then 0; validation on 96 cases over every family.
# Sequential: 4 layers, 10 layers, then the scale control (4 layers at the Milestone 10 budget). Waits for the bases.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
until [ -f runs/m12/base_l4-s0/final.pt ] && [ -f runs/m12/base_l10-s0/final.pt ]; do sleep 120; done
common=(--gains runs/m9/gains_shared.json --gain-residual --ext-torque --free-teacher ideal --oracle-f_min 7 --free-weight 3 --contact-weight 1 --families consolidation --dist v5
        --expand-inputs --val-families all --val-cases 4 --val-cases-contact 8 --max-rows 0 --workers 8 --threads 2 --seed 0)
scale=(--iters 20 --cases 320 --betas 1,0.5,0.5,0.5,0.5,0.25,0.25,0.25,0.25,0)
uv run --no-sync python experiments/d7_distill.py $common $scale --warmstart runs/m12/base_l4-s0/best.pt --variant consol_l4 --out runs/m12/student_l4-s0 > runs/m12/student_l4-s0.log 2>&1
uv run --no-sync python experiments/d7_distill.py $common $scale --warmstart runs/m12/base_l10-s0/best.pt --variant consol_l10 --out runs/m12/student_l10-s0 > runs/m12/student_l10-s0.log 2>&1
uv run --no-sync python experiments/d7_distill.py $common --iters 12 --cases 48 --betas 1,0.5,0.5,0.5,0 --warmstart runs/m12/base_l4-s0/best.pt --variant consol_l4_small --out runs/m12/student_l4_small-s0 > runs/m12/student_l4_small-s0.log 2>&1
echo "phase-2 students done at $(date +%H:%M)"
