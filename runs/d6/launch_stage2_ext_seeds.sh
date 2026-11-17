#!/bin/zsh
# Contact deep dive D5, stage 2: seeds 1 and 2 of the external-torque PPO arms (seed 0 beat the warm start at its first validation).
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=.
export LLC_VAL_WORKERS=4
WS=runs/d4/pd_residual_acmed_s02_lowrate-s0
PPO=(--warmstart $WS/best.pt --critic $WS/critic.pt --families contact_focus --start-mode mixed --hold 10 --horizon 4.0 --updates 200 --envs 64
     --val-every 25 --val-cases 2 --val-cases-contact 8 --val-objective composite_cap --reflex-force 25 --reflex-tau 0.02 --no-reflex-terminate
     --noise-rho 0.5 --init-std 0.05 --threads 1 --sim-threads 2 --pool 48 --w-press 1.0 --force-cap 15 --force-tau 0.02 --ext-torque)
run() { local name=$1; shift; mkdir -p runs/d6/$name; nohup uv run --no-sync python "$@" --out runs/d6/$name > runs/d6/$name/train.log 2>&1 & }
for s in 1 2; do
  run ppo_cap_gain_ext-s$s experiments/d5_ppo.py "${PPO[@]}" --gain-residual --seed $s --variant cap_gain_ext
  run ppo_cap_torque_ext-s$s experiments/d5_ppo.py "${PPO[@]}" --seed $s --variant cap_torque_ext
done
echo "launched 4 external-torque PPO seeds at $(date +%H:%M)"
wait
echo "external-torque PPO seeds finished at $(date +%H:%M)"
