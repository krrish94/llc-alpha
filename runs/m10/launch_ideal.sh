#!/bin/zsh
# Milestone 10: the TAM-style ideal-robot residual on plant v3, three seeds (three DAgger-style iterations of 48 free-space episodes).
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3
for s in 0 1 2; do
  nohup uv run --no-sync python experiments/m10_ideal.py --gains runs/m9/gains_shared.json --iters 3 --cases 48 --workers 3 --threads 2 --seed $s --out runs/m10/ideal-s$s > runs/m10/ideal-s$s.log 2>&1 &
done
echo "ideal residual seeds launched at $(date +%H:%M)"
