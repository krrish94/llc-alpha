#!/bin/zsh
# Contact deep dive D5, stage 2 (early, seed 0): the contact-sensing input (external-torque estimate) for both methods and both
# parameterisations, and the wider-exploration PPO arms; launched while stage 1 runs because stage 1's first validations show
# both methods trading free-space tracking for small yields instead of becoming compliant only in contact.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=.
export LLC_VAL_WORKERS=4
WS=runs/d4/pd_residual_acmed_s02_lowrate-s0
AC=(--objective ac --grad fd --force-grad detach --median-clip 3.0 --pos-sigma 0.02 --w-rate 0.003 --init-from $WS/best.pt --mix contact --start-mode mixed
    --stage-iters 150 150 150 --batch 24 --horizon 2.0 --val-every 50 --val-cases 2 --val-cases-contact 8 --val-objective composite_cap
    --pool 48 --threads 1 --sim-threads 2 --keep-checkpoints --force-tau 0.02 --lr 1e-4 --w-press 1.0 --force-cap 15)
PPO=(--warmstart $WS/best.pt --critic $WS/critic.pt --families contact_focus --start-mode mixed --hold 10 --horizon 4.0 --updates 200 --envs 64
     --val-every 25 --val-cases 2 --val-cases-contact 8 --val-objective composite_cap --reflex-force 25 --reflex-tau 0.02 --no-reflex-terminate
     --noise-rho 0.5 --threads 1 --sim-threads 2 --pool 48 --w-press 1.0 --force-cap 15 --force-tau 0.02)
run() { local name=$1; shift; mkdir -p runs/d6/$name; nohup uv run --no-sync python "$@" --out runs/d6/$name > runs/d6/$name/train.log 2>&1 & }
run ac_cap_gain_ext-s0 experiments/d2_train.py --kind pd_gain_residual "${AC[@]}" --ext-torque --seed 0 --variant cap_gain_ext
run ac_cap_torque_ext-s0 experiments/d2_train.py --kind pd_residual "${AC[@]}" --ext-torque --seed 0 --variant cap_torque_ext
run ppo_cap_gain_ext-s0 experiments/d5_ppo.py "${PPO[@]}" --gain-residual --ext-torque --init-std 0.05 --seed 0 --variant cap_gain_ext
run ppo_cap_torque_ext-s0 experiments/d5_ppo.py "${PPO[@]}" --ext-torque --init-std 0.05 --seed 0 --variant cap_torque_ext
run ppo_cap_gain_explore-s0 experiments/d5_ppo.py "${PPO[@]}" --gain-residual --init-std 0.15 --lr-actor 3e-4 --seed 0 --variant cap_gain_explore
run ppo_cap_torque_explore-s0 experiments/d5_ppo.py "${PPO[@]}" --init-std 0.15 --lr-actor 3e-4 --seed 0 --variant cap_torque_explore
echo "launched 6 early stage-2 runs at $(date +%H:%M)"
wait
echo "early stage-2 runs finished at $(date +%H:%M)"
