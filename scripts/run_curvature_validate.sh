#!/usr/bin/env bash
# Mechanism check: is exact neutral-lift Tweedie curvature a useful signal?
# The exact ranking equals Gini impurity here; the report includes both names
# and retains the former sensitivity statistic as an explicit ablation.
set -euo pipefail
cd "$(dirname "$0")/.."
CHECKPOINT="${CHECKPOINT:-checkpoints/mdlm.ckpt}"
ARCHITECTURE="${ARCHITECTURE:-small}"
DATA_SOURCE="${DATA_SOURCE:-openwebtext}"
OUTPUT="${OUTPUT:-curvature_validate_results.json}"

# Preflight: a missing checkpoint used to crash the run *after* an old JSON was
# already on disk, so a stale smoke result masqueraded as a fresh one. Fail loud
# and delete any stale output up front so a crash leaves no result behind.
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[FAIL] checkpoint not found: $CHECKPOINT" >&2
  echo "       Set CHECKPOINT=/abs/path/to/mdlm.ckpt (the gate needs a trained" >&2
  echo "       MDLM; without it there is nothing to validate)." >&2
  exit 1
fi
sz=$(stat -c%s "$CHECKPOINT" 2>/dev/null || stat -f%z "$CHECKPOINT")
if [[ "${sz:-0}" -lt 10000000 ]]; then
  echo "[FAIL] checkpoint suspiciously small (${sz} bytes): $CHECKPOINT" >&2
  exit 1
fi
echo "[preflight] checkpoint ok ($((sz/1024/1024)) MB): $CHECKPOINT"
echo "[preflight] n_states=${N_STATES:-128} n_probes=${N_PROBES:-8} arch=$ARCHITECTURE"
rm -f "$OUTPUT"  # never let a crash leave a stale result

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
  --output "$OUTPUT"

echo "[done] wrote $OUTPUT"
