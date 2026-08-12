#!/usr/bin/env bash
# E3: exact Gumbel-Tweedie curvature-aware decoding.
# Order parallel unmasking by posterior noise-score covariance.
#
# Usage:
#   CHECKPOINT=checkpoints/mdlm.ckpt bash scripts/run_curvature_decode.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT="${CHECKPOINT:-checkpoints/mdlm.ckpt}"
ARCHITECTURE="${ARCHITECTURE:-small}"
ORDERS="${ORDERS:-confidence entropy curvature random}"
STEPS="${STEPS:-4 8 16 32}"
N_SAMPLES="${N_SAMPLES:-512}"
SIGNAL_SCALE="${SIGNAL_SCALE:-1.0}"
GUMBEL_SCALE="${GUMBEL_SCALE:-1.0}"
N_PROBES="${N_PROBES:-4}"
IN_SCALE="${IN_SCALE:-6.0}"
EPS="${EPS:-0.1}"
JUDGE="${JUDGE:-gpt2-large}"

# Exact curvature needs no probes and costs no extra NFE. N_PROBES, IN_SCALE,
# and EPS are used only when ORDERS includes the finite-difference
# `sensitivity` ablation.
python -u curvature_decode.py \
  --model ckpt --checkpoint "$CHECKPOINT" \
  --architecture "$ARCHITECTURE" \
  --orders ${ORDERS} \
  --steps ${STEPS} \
  --n-samples "$N_SAMPLES" \
  --signal-scale "$SIGNAL_SCALE" \
  --gumbel-scale "$GUMBEL_SCALE" \
  --n-probes "$N_PROBES" \
  --in-scale "$IN_SCALE" \
  --eps "$EPS" \
  --judge "$JUDGE" \
  --output curvature_decode_results.json
