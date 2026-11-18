#!/bin/zsh
# D6: final test-protocol evaluation (stage 1 and stage 2 arms, baselines, oracles, reference arms); episodes saved for the figures and clips.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=.
uv run --no-sync python experiments/d_evaluate.py --feedforward --oracle-on pd runs/d4/pd_residual_acmed_s02_lowrate-s0 \
  --runs runs/d4/pd_residual_acmed_s02_lowrate-s0 runs/d5/ppo_contact_v2-s0@final.pt runs/d6/ppo_cap_gain_ext-s0 runs/d6/ppo_cap_gain_ext-s1 runs/d6/ppo_cap_gain_ext-s2 \
         runs/d7/distill_gain_ext-s0 runs/d7/distill_gain_ext-s1 runs/d7/distill_gain_ext-s2 \
         runs/d7/distill_gain_ext_preview-s0 runs/d7/distill_gain_ext_preview-s1 runs/d7/distill_gain_ext_preview-s2 \
         runs/d7/ppo_distill_gain_ext-s0 runs/d7/ppo_distill_gain_ext-s1 runs/d7/ppo_distill_gain_ext-s2 runs/d7/ppo_cap_gain_ext_anchor-s0 \
  --out runs/d7/metrics_test.json --save-episodes runs/d7/episodes_test "$@"
