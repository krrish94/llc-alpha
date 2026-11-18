#!/bin/zsh
# Milestone 11, Phase 2e (ADR 0011 amendment 3, item 4): the P1 student's recipe with the teacher's share held at 0.5 for three more
# iterations (the Phase 2c variant that kept free space and took the blocked family closest to the oracle) in seeds 1 and 2, and
# the same recipe with eight layers (decision 6); then the evaluation of the three seeds pooled and the eight-layer run through P1
# next to the classical set (the re-tuned PD included) and the oracle, and the extended probe of seed 0. Waits for the force-term runs.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
until [ -f runs/m11/student_p1_f1-s0/final.pt ]; do sleep 60; done
common=(--profile p1 --gain-residual --ext-torque --free-teacher ideal --teacher-base pd --gains runs/m9/gains_shared.json --oracle-f_min 7
        --iters 12 --cases 48 --workers 4 --threads 2 --hidden 256 --betas 1,0.5,0.5,0.5,0 --free-weight 3 --contact-weight 1)
for s in 1 2; do
  uv run --no-sync python experiments/d7_distill.py $common --layers 4 --seed $s --variant host_p1_beta --out runs/m11/student_p1_beta-s$s > runs/m11/student_p1_beta-s$s.log 2>&1
done
uv run --no-sync python experiments/d7_distill.py $common --layers 8 --seed 0 --variant host_p1_beta_l8 --out runs/m11/student_p1_beta_l8-s0 > runs/m11/student_p1_beta_l8-s0.log 2>&1
uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains pd_shared_cap=runs/m10/gains_shared.json pd_retuned=runs/m11/tune_p1/gains_shared.json --feedforward --oracle-on pd --workers 10 \
  --profile p1 --runs runs/m11/student_p1-s0 runs/m11/student_p1_beta-s0 runs/m11/student_p1_beta-s1 runs/m11/student_p1_beta-s2 runs/m11/student_p1_beta_l8-s0 \
  --out runs/m11/metrics_phase2e_p1.json --save-episodes runs/m11/episodes_phase2e_p1 > runs/m11/metrics_phase2e_p1.log 2>&1
uv run --no-sync python experiments/m11_probe.py --profile p1 --cases 8 --workers 4 --threads 2 --out runs/m11/probe_ext_student_p1_beta.json student_p1_beta=runs/m11/student_p1_beta-s0/best.pt > runs/m11/probe_ext_student_p1_beta.log 2>&1
echo "phase 2e done at $(date +%H:%M)"
