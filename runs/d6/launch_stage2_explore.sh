#!/bin/zsh
# Contact deep dive D5, stage 2b: PPO exploration/optimiser arms on the cap reward (wider exploration, faster actor), both parameterisations, three seeds.
# Usage: zsh runs/d6/launch_stage2_explore.sh
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=.
export LLC_VAL_WORKERS=4
WS=runs/d4/pd_residual_acmed_s02_lowrate-s0
PPO=(--warmstart $WS/best.pt --critic $WS/critic.pt --families contact_focus --start-mode mixed --hold 10 --horizon 4.0 --updates 200 --envs 64
     --val-every 25 --val-cases 2 --val-cases-contact 8 --val-objective composite_cap --reflex-force 25 --reflex-tau 0.02 --no-reflex-terminate
     --noise-rho 0.5 --init-std 0.15 --lr-actor 3e-4 --threads 1 --sim-threads 2 --pool 48 --w-press 1.0 --force-cap 15 --force-tau 0.02)
run() { local name=$1; shift; mkdir -p runs/d6/$name; nohup uv run --no-sync python "$@" --out runs/d6/$name > runs/d6/$name/train.log 2>&1 & }
for s in 0 1 2; do
  run ppo_cap_torque_explore-s$s experiments/d5_ppo.py "${PPO[@]}" --seed $s --variant cap_torque_explore
  run ppo_cap_gain_explore-s$s experiments/d5_ppo.py "${PPO[@]}" --gain-residual --seed $s --variant cap_gain_explore
done
echo "launched 6 exploration arms at $(date +%H:%M)"
wait
echo "exploration arms finished at $(date +%H:%M)"
