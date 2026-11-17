#!/bin/zsh
# Milestone 9: light-contact detachment diagnostic, pads (v3, re-tuned gains) against the sphere (v2, its gains).
cd "$(dirname "$0")/../.."  # the repository root
export PYTHONPATH=.
LLC_PLANT=v3 uv run --no-sync python experiments/m9_detachment.py --gains runs/m9/gains_shared.json --out runs/m9/detachment_v3.json "$@"
LLC_PLANT=v2 uv run --no-sync python experiments/m9_detachment.py --gains runs/d1/gains_shared.json --out runs/m9/detachment_v2.json "$@"
