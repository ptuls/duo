#!/usr/bin/env bash
# Token-level A_i characterization on a real MDLM. Resolves the char-level caveat:
# how large is the plausible support on a 50k vocab, and how big must A_i be?
set -euo pipefail
cd "$(dirname "$0")/.."
CHECKPOINT="${CHECKPOINT:-checkpoints/mdlm.ckpt}"
python -u characterize_Ai_token.py \
  --model ckpt --checkpoint "$CHECKPOINT" \
  --n-seqs "${N_SEQS:-128}" \
  --seq-len "${SEQ_LEN:-128}" \
  --mask-frac "${MASK_FRAC:-0.5}" \
  --max-candidates "${MAX_CANDIDATES:-16}" \
  --ms ${MS:-8 32 128 512} \
  --bits ${BITS:-3} \
  --output characterize_Ai_token.json
