#!/bin/zsh
# Milestone 12, phase 1 of the two-phase student (ADR 0012, decision 5): the free-space residual base trained through the plant by the
# Milestone 10 round-2 recipe (actor-critic, bounded kernels, median clipping, the inverse-dynamics feedforward in the PD base, selected
# on the free-space families), at the nominal base gains, for the two depths: 4 layers of 256 and 10 layers of 256 with residual blocks
# and layer normalisation. Seed 0, both at once (2 torch threads + 3 simulator threads each).
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3 LLC_VAL_WORKERS=3
for depth in 4 10; do
  out=runs/m12/base_l$depth-s0
  mkdir -p $out
  extra=""; [[ $depth == 10 ]] && extra="--residual-blocks"
  nohup uv run --no-sync python experiments/d2_train.py --kind pd_residual --feedforward --gains runs/m9/gains_shared.json --objective ac --ac-horizon 100 --median-clip 3 --pos-sigma 0.02 --w-rate 0.003 \
    --lr 1e-3 --stage-iters 100 150 0 --batch 24 --horizon 2.0 --val-every 25 --val-cases 4 --val-cases-contact 2 --val-objective free --keep-checkpoints --threads 2 --sim-threads 3 \
    --layers $depth --hidden 256 $extra --seed 0 --variant consol_l$depth --out $out > $out/train.log 2>&1 &
done
echo "phase-1 bases launched at $(date +%H:%M)"
