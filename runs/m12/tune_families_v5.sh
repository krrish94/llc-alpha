#!/bin/zsh
# Milestone 12 (ADR 0012): the per-family PD of the five protocol v5 families (wipe, erase, peg, drawer, door), the same CMA-ES
# budget and objective as tune_families.sh; waits for that chain to finish.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
until grep -q "done at" runs/m12/tune_families.log 2>/dev/null; do sleep 60; done
for fam in wipe erase peg drawer door; do
  uv run --no-sync python experiments/m1_tune_gains.py --target $fam --protocol v2 --objective composite_cap --feedforward --n-cases 4 --iters 40 --popsize 12 --workers 6 \
    --out-dir runs/m12/tune_$fam > runs/m12/tune_$fam.log 2>&1
done
echo "per-family tunes (v5 families) done at $(date +%H:%M)"
