#!/bin/zsh
# D6 stage 2: PPO under the cap with the stop-and-wait anchor, from the distilled students (three seeds) and the control arm
# (the D5 external-torque PPO arm resumed with the anchored reward only). Reward press target 500 N/m (ADR 0008).
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=.
export LLC_VAL_WORKERS=4
CRITIC=runs/d6/ppo_cap_gain_ext-s0/critic.pt
PPO=(--families contact_focus --start-mode mixed --hold 10 --horizon 4.0 --updates 150 --envs 64
     --val-every 25 --val-cases 2 --val-cases-contact 8 --val-objective composite_cap --reflex-force 25 --reflex-tau 0.02 --no-reflex-terminate
     --noise-rho 0.5 --init-std 0.05 --threads 1 --sim-threads 2 --pool 48 --w-press 1.0 --force-cap 15 --force-tau 0.02 --press-stiffness 500 --ext-torque --gain-residual)
run() { local name=$1; shift; mkdir -p runs/d7/$name; nohup uv run --no-sync python "$@" --out runs/d7/$name > runs/d7/$name/train.log 2>&1 & }
for s in 0 1 2; do
  run ppo_distill_gain_ext-s$s experiments/d5_ppo.py "${PPO[@]}" --warmstart runs/d7/distill_gain_ext-s$s/best.pt --critic $CRITIC --seed $s --variant distill_gain_ext
done
run ppo_cap_gain_ext_anchor-s0 experiments/d5_ppo.py "${PPO[@]}" --warmstart runs/d6/ppo_cap_gain_ext-s0/last.pt --critic $CRITIC --seed 0 --variant cap_gain_ext_anchor
echo "launched stage 2 at $(date +%H:%M)"
