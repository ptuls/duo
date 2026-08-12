#!/usr/bin/env bash
# Mechanism check: is posterior curvature a real signal? Run this BEFORE the
# decoding experiment. If curvature does not beat entropy, margin, and its own
# shuffled null on the stability target, a decoding win cannot be attributed to
# the theory -- do not proceed to run_curvature_decode.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
CHECKPOINT="${CHECKPOINT:-checkpoints/mdlm.ckpt}"
python -u curvature_validate.py \
  --model ckpt --checkpoint "$CHECKPOINT" \
  --n-states "${N_STATES:-128}" \
  --seq-len "${SEQ_LEN:-128}" \
  --mask-frac "${MASK_FRAC:-0.5}" \
  --max-candidates "${MAX_CANDIDATES:-32}" \
  --n-probes "${N_PROBES:-8}" \
  --in-scale "${IN_SCALE:-6.0}" \
  --output curvature_validate_results.json
