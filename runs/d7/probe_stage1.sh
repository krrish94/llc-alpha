#!/bin/zsh
# D6: contact-sensing metrics (linear probe on the trunk features) of the warm start, the D5 external-torque arm and the students.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=.
uv run --no-sync python experiments/d_contact_probe.py --out runs/d7/probe.json --threads 4 \
  pd_residual_acmed_s02_lowrate=runs/d4/pd_residual_acmed_s02_lowrate-s0/best.pt ppo_cap_gain_ext=runs/d6/ppo_cap_gain_ext-s0/best.pt \
  distill_gain_ext=runs/d7/distill_gain_ext-s0/best.pt distill_gain_ext_preview=runs/d7/distill_gain_ext_preview-s0/best.pt "$@"
