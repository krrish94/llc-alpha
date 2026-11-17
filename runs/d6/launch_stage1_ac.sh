#!/bin/zsh
# Contact deep dive D5, stage 1, actor-critic arms with the force terms detached from the gradient (ADR 0007 amendment).
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=.
export LLC_VAL_WORKERS=4
WS=runs/d4/pd_residual_acmed_s02_lowrate-s0
AC=(--objective ac --grad fd --force-grad detach --median-clip 3.0 --pos-sigma 0.02 --w-rate 0.003 --init-from $WS/best.pt --mix contact --start-mode mixed
    --stage-iters 150 150 150 --batch 24 --horizon 2.0 --val-every 50 --val-cases 2 --val-cases-contact 8 --val-objective composite_cap
    --pool 48 --threads 1 --sim-threads 2 --keep-checkpoints --force-tau 0.02 --lr 1e-4)
CAP=(--w-press 1.0 --force-cap 15)
V2=(--force-soft 30)
run() { local name=$1; shift; mkdir -p runs/d6/$name; nohup uv run --no-sync python "$@" --out runs/d6/$name > runs/d6/$name/train.log 2>&1 & }
for s in 0 1 2; do
  run ac_cap_torque-s$s experiments/d2_train.py --kind pd_residual "${AC[@]}" "${CAP[@]}" --seed $s --variant cap_torque
  run ac_cap_gain-s$s experiments/d2_train.py --kind pd_gain_residual "${AC[@]}" "${CAP[@]}" --seed $s --variant cap_gain
done
run ac_v2_torque-s0 experiments/d2_train.py --kind pd_residual "${AC[@]}" "${V2[@]}" --seed 0 --variant v2_torque
run ac_v2_gain-s0 experiments/d2_train.py --kind pd_gain_residual "${AC[@]}" "${V2[@]}" --seed 0 --variant v2_gain
echo "launched 8 actor-critic runs at $(date +%H:%M)"
wait
echo "actor-critic arms finished at $(date +%H:%M)"
