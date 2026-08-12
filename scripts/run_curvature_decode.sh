#!/usr/bin/env bash
# E3: curvature-aware decoding (the level-two flagship).
# Order parallel unmasking by posterior curvature vs confidence/entropy/random.
#
# Usage:
#   CHECKPOINT=checkpoints/mdlm.ckpt bash scripts/run_curvature_decode.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT="${CHECKPOINT:-checkpoints/mdlm.ckpt}"
ORDERS="${ORDERS:-confidence entropy curvature random}"
STEPS="${STEPS:-4 8 16 32}"
N_SAMPLES="${N_SAMPLES:-512}"
N_PROBES="${N_PROBES:-4}"
IN_SCALE="${IN_SCALE:-6.0}"
EPS="${EPS:-0.1}"
JUDGE="${JUDGE:-gpt2-large}"

# CALIBRATION FIRST: run a tiny sweep and confirm curvature scores have spread
# (not all near-equal, which means in_scale is saturating). Then run the full
# comparison. The decisive contrast is curvature vs entropy: both defer
# uncertain tokens, but only curvature uses the second-order signal.
python -u curvature_decode.py \
  --model ckpt --checkpoint "$CHECKPOINT" \
  --orders ${ORDERS} \
  --steps ${STEPS} \
  --n-samples "$N_SAMPLES" \
  --n-probes "$N_PROBES" \
  --in-scale "$IN_SCALE" \
  --eps "$EPS" \
  --judge "$JUDGE" \
  --output curvature_decode_results.json
