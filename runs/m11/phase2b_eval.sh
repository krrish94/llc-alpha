#!/bin/zsh
# Milestone 11, Phase 2b: evaluate each host-rate student (as it finishes) through its profile next to the classical set and the
# oracle through the same profile, on every group of protocol v4 (the torque-tracking group runs the torque program), and probe
# its trunk layer by layer through the profile. Waits for final.pt of each student run.
cd "$(dirname "$0")/../.."
export PYTHONPATH=. LLC_PLANT=v3
for prof in p1 p2 p3; do
  until [ -f runs/m11/student_$prof-s0/final.pt ]; do sleep 60; done
  uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains pd_shared_cap=runs/m10/gains_shared.json --feedforward --oracle-on pd --workers 10 \
    --profile $prof --runs runs/m11/student_$prof-s0 --out runs/m11/metrics_students_$prof.json --save-episodes runs/m11/episodes_students_$prof > runs/m11/metrics_students_$prof.log 2>&1
  uv run --no-sync python experiments/m11_probe.py --profile $prof --out runs/m11/probe_student_$prof.json --workers 4 --threads 2 student_$prof=runs/m11/student_$prof-s0/best.pt > runs/m11/probe_student_$prof.log 2>&1
done
echo "phase 2b evaluations done at $(date +%H:%M)"
