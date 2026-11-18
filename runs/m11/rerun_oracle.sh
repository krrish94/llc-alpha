#!/bin/zsh
# Milestone 11, Phase 1: the oracle ceiling re-evaluated through profiles P1 to P3 after the time-indexing fix of its frame path
# (2026-11-17; it had advanced its reference once per host frame instead of once per millisecond); the classical rows of
# metrics_p*.json are kept and the oracle_pd rows replaced.
cd "$(dirname "$0")/../.."
export PYTHONPATH=. LLC_PLANT=v3
for prof in p1 p2 p3; do
  uv run --no-sync python experiments/d_evaluate.py --gains runs/m9/gains_shared.json --extra-gains --oracle-on pd --workers 12 --profile $prof --groups in_distribution tasks surface_500 --out runs/m11/oracle_$prof.json
  python3 - "$prof" <<'PY'
import json, sys
prof = sys.argv[1]
main = json.load(open(f"runs/m11/metrics_{prof}.json")); new = json.load(open(f"runs/m11/oracle_{prof}.json"))
for g, res in new["groups"].items():
    if "oracle_pd" in res:
        main["groups"][g]["oracle_pd"] = res["oracle_pd"]
main["oracle_rerun"] = "oracle_pd rows re-evaluated after the oracle's time-indexing fix (2026-11-17)"
json.dump(main, open(f"runs/m11/metrics_{prof}.json", "w"), indent=1)
print("merged", prof)
PY
done
echo "oracle rerun done at $(date +%H:%M)"
