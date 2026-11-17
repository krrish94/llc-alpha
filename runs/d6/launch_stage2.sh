#!/bin/zsh
# Contact deep dive D5, stage 2 (ADR 0007): on the better parameterisation of stage 1, the model study:
# external-torque input, Fourier features, 20 ms output filter; actor-critic and PPO, three seeds each.
# Usage: zsh runs/d6/launch_stage2.sh <torque|gain>
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=.
export LLC_VAL_WORKERS=4
PARAM=${1:-gain}
if [[ $PARAM == gain ]]; then KIND=pd_gain_residual; GAINFLAG=(--gain-residual); else KIND=pd_residual; GAINFLAG=(); fi
WS=runs/d4/pd_residual_acmed_s02_lowrate-s0
AC_BASE=(--objective ac --grad fd --force-grad detach --median-clip 3.0 --pos-sigma 0.02 --w-rate 0.003 --mix contact --start-mode mixed
    --stage-iters 150 150 150 --batch 24 --horizon 2.0 --val-every 50 --val-cases 2 --val-cases-contact 8 --val-objective composite_cap
    --pool 48 --threads 1 --sim-threads 2 --keep-checkpoints --force-tau 0.02 --w-press 1.0 --force-cap 15 --lr 1e-4)
AC=(--init-from $WS/best.pt "${AC_BASE[@]}")
PPO=(--warmstart $WS/best.pt --critic $WS/critic.pt --families contact_focus --start-mode mixed --hold 10 --horizon 4.0 --updates 200 --envs 64
     --val-every 25 --val-cases 2 --val-cases-contact 8 --val-objective composite_cap --reflex-force 25 --reflex-tau 0.02 --no-reflex-terminate
     --noise-rho 0.5 --init-std 0.05 --force-tau 0.02 --threads 1 --sim-threads 2 --pool 48 --w-press 1.0 --force-cap 15)
run() { local name=$1; shift; mkdir -p runs/d6/$name; nohup uv run --no-sync python "$@" --out runs/d6/$name > runs/d6/$name/train.log 2>&1 & }
for s in 0 1 2; do
  run ac_cap_${PARAM}_ext-s$s experiments/d2_train.py --kind $KIND "${AC[@]}" --ext-torque --seed $s --variant cap_${PARAM}_ext
  run ac_cap_${PARAM}_filt-s$s experiments/d2_train.py --kind $KIND "${AC[@]}" --output-tau 0.02 --seed $s --variant cap_${PARAM}_filt
  run ppo_cap_${PARAM}_ext-s$s experiments/d5_ppo.py "${PPO[@]}" "${GAINFLAG[@]}" --ext-torque --seed $s --variant cap_${PARAM}_ext
  run ppo_cap_${PARAM}_filt-s$s experiments/d5_ppo.py "${PPO[@]}" "${GAINFLAG[@]}" --output-tau 0.02 --seed $s --variant cap_${PARAM}_filt
done
# Fourier features cannot expand a raw-encoded warm start exactly: the actor-critic arm trains the encoding from the PD base (no init), one seed per parameterisation
run ac_cap_${PARAM}_fourier-s0 experiments/d2_train.py --kind $KIND "${AC_BASE[@]}" --encoding fourier --seed 0 --variant cap_${PARAM}_fourier
echo "launched stage 2 ($PARAM) at $(date +%H:%M)"
wait
echo "stage 2 finished at $(date +%H:%M)"
