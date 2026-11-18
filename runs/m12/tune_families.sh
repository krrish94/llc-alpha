#!/bin/zsh
# Milestone 12 (ADR 0012, decision 1): the per-family PD, CMA-ES per family on the cap composite in the feedforward form, the
# Milestone 9 budget (40 iterations of 12 candidates on four tune cases), six workers so that the other agent's runs keep their share.
# The nine families of protocols v3 and v4 first; the new families of protocol v5 run from tune_families_v5.sh once they exist.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
for fam in hold free policy step contact2 blocked task push pick; do
  uv run --no-sync python experiments/m1_tune_gains.py --target $fam --protocol v2 --objective composite_cap --feedforward --n-cases 4 --iters 40 --popsize 12 --workers 6 \
    --out-dir runs/m12/tune_$fam > runs/m12/tune_$fam.log 2>&1
done
echo "per-family tunes (v3/v4 families) done at $(date +%H:%M)"
