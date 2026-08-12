#!/usr/bin/env bash
# Mechanism check: is exact neutral-lift Tweedie curvature a useful signal?
# The exact ranking equals Gini impurity here; the report includes both names
# and retains the former sensitivity statistic as an explicit ablation.
set -euo pipefail
cd "$(dirname "$0")/.."
CHECKPOINT="${CHECKPOINT:-checkpoints/mdlm.ckpt}"
ARCHITECTURE="${ARCHITECTURE:-small}"
DATA_SOURCE="${DATA_SOURCE:-openwebtext}"
python -u curvature_validate.py \
  --model ckpt --checkpoint "$CHECKPOINT" \
  --architecture "$ARCHITECTURE" \
  --data-source "$DATA_SOURCE" \
  --n-states "${N_STATES:-128}" \
  --seq-len "${SEQ_LEN:-128}" \
  --mask-frac "${MASK_FRAC:-0.5}" \
  --max-candidates "${MAX_CANDIDATES:-32}" \
  --signal-scale "${SIGNAL_SCALE:-1.0}" \
  --gumbel-scale "${GUMBEL_SCALE:-1.0}" \
  --n-probes "${N_PROBES:-8}" \
  --in-scale "${IN_SCALE:-6.0}" \
  --eps "${EPS:-0.1}" \
  --output curvature_validate_results.json
