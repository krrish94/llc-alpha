#!/bin/zsh
# D6: test-protocol evaluation after stage 1 (students), with the baselines, the oracles and the D3/D5 reference arms; episodes saved for the figures and clips.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=.
uv run --no-sync python experiments/d_evaluate.py --feedforward --oracle-on pd runs/d4/pd_residual_acmed_s02_lowrate-s0 \
  --runs runs/d4/pd_residual_acmed_s02_lowrate-s0 runs/d5/ppo_contact_v2-s0@final.pt runs/d6/ppo_cap_gain_ext-s0 runs/d6/ppo_cap_gain_ext-s1 runs/d6/ppo_cap_gain_ext-s2 \
         runs/d7/distill_gain_ext-s0 runs/d7/distill_gain_ext-s1 runs/d7/distill_gain_ext-s2 \
         runs/d7/distill_gain_ext_preview-s0 runs/d7/distill_gain_ext_preview-s1 runs/d7/distill_gain_ext_preview-s2 \
  --out runs/d7/metrics_stage1.json --save-episodes runs/d7/episodes_stage1 "$@"
