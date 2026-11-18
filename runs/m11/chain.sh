#!/bin/zsh
# Milestone 11 chain after the interface fix of 2026-11-17: Phase 1 evaluations and sweep, the YAM gain tune, the host-rate students, the Panda re-tunes.
cd "$(dirname "$0")/../.."
./runs/m11/phase1_eval.sh > runs/m11/phase1_eval.log 2>&1
./runs/m11/yam/tune_p1.sh
./runs/m11/phase2_students.sh 0 > runs/m11/phase2_students.log 2>&1
./runs/m11/phase1_tune.sh > runs/m11/phase1_tune.log 2>&1
echo "chain done at $(date +%H:%M)"
