#!/bin/zsh
# D6: contact-sensing metrics of the stage-2 arms, appended to the stage-1 probe.
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=.
uv run --no-sync python experiments/d_contact_probe.py --out runs/d7/probe_stage2.json --threads 4 \
  ppo_distill_gain_ext=runs/d7/ppo_distill_gain_ext-s0/best.pt ppo_cap_gain_ext_anchor=runs/d7/ppo_cap_gain_ext_anchor-s0/best.pt "$@"
