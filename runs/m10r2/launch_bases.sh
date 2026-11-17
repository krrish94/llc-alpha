#!/bin/zsh
# Milestone 10, round 2 (ADR 0010 amendment): the bases of the distilled students with the inverse-dynamics feedforward in the
# PD base (A), trained on the free-space families only and selected on them (B); a PD residual and a direct torque network
# (warm-started on the tuned PD + feedforward law), three seeds each. D3 recipe otherwise (actor-critic through the plant,
# bounded kernels, median clipping), v3 gains, plant v3. Six runs at once: 2 torch threads + 3 simulator threads each.
# The torque network takes a tenfold smaller learning rate: at 1e-3 the first actor-critic updates destroyed the warm start
# (tracking kernel 0.70 to 0.05-0.2, impacts of 4-5 kN on the table within five iterations); a residual on a PD base tolerates 1e-3.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=. LLC_PLANT=v3 LLC_VAL_WORKERS=3
for kind in ${KINDS:-pd_residual neural_torque}; do
  lr=1e-3; [[ $kind == neural_torque ]] && lr=1e-4
  for s in 0 1 2; do
    out=runs/m10r2/${kind}_ff-s$s
    mkdir -p $out
    nohup uv run --no-sync python experiments/d2_train.py --kind $kind --feedforward --gains runs/m9/gains_shared.json --objective ac --ac-horizon 100 --median-clip 3 --pos-sigma 0.02 --w-rate 0.003 \
      --lr $lr --stage-iters 100 150 0 --batch 24 --horizon 2.0 --val-every 25 --val-cases 4 --val-cases-contact 2 --val-objective free --keep-checkpoints --threads 2 --sim-threads 3 \
      --seed $s --variant v3_ff --out $out > $out/train.log 2>&1 &
  done
done
echo "bases launched at $(date +%H:%M)"
