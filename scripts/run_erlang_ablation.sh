#!/bin/bash
# Erlang-k semi-Markov masking ablation (algo=erlang_mdlm).
#
# Trains the phase-augmented masked diffusion model from scratch on OpenWebText
# for k in {1, 2, 4} under identical settings and logs generative perplexity.
# k=1 has the plain MDLM forward process and objective. The runs share the
# time-conditioned architecture, so the ablation isolates whether observable
# Erlang phase improves hard-state generation over the k=1 process. Does it lower
# gen_ppl over hard binary masking?
#
# The claim under test is the semi-Markov appendix result. The analytic phase
# gate (diffusion-algorithimic-information-theory/experiments/gumbel_lift/
# semimarkov_phase_gate.py) showed the phase carries decision-relevant signal
# only for k>1; this is the real-model version of that test.
#
# No srun: DDP is launched via torch device_count (the box lacks slurm).
# Forwards extra hydra overrides via "$@" (e.g. trainer.max_steps=50000).
#
# Usage:
#   DUO_DATA_DIR=/writable/path bash scripts/run_erlang_ablation.sh
#   KS="1 2 4 8" STEPS=100000 BATCH_SIZE=8 bash scripts/run_erlang_ablation.sh

set -euo pipefail
export HYDRA_FULL_ERROR=1

KS=${KS:-"1 2 4"}
STEPS=${STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-8}
MODEL=${MODEL:-small}
LENGTH=${LENGTH:-1024}
DATA=${DATA:-openwebtext-split}
# W&B: offline by default (no network/permission surprises mid-run). Set
# WANDB_OFFLINE=False to stream live to wandb.ai; requires `wandb login` on the
# box. entity=null in config.yaml uses your personal entity (avoids the org
# "Create Run permission" error). Override project/entity via "$@" if needed,
# e.g. wandb.project=erlang wandb.entity=<user>.
WANDB_OFFLINE=${WANDB_OFFLINE:-True}
export DUO_DATA_DIR=${DUO_DATA_DIR:-$PWD/data}
mkdir -p "$DUO_DATA_DIR" watch_folder

echo "[erlang] ablation over k in: $KS"
echo "[erlang] data=$DATA steps=$STEPS batch=$BATCH_SIZE model=$MODEL length=$LENGTH"
echo "[erlang] data cache: $DUO_DATA_DIR  wandb offline: $WANDB_OFFLINE"

# ------------------------------------------------------------------ preflight
# Fail fast rather than a traceback minutes in, or a silent multi-hour
# OpenWebText re-tokenization. Skip with SKIP_PREFLIGHT=1.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  echo "[preflight] checking launch prerequisites..."
  ok=1
  # 1. GPU visible (this is a from-scratch training run, not CPU-viable).
  if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  [FAIL] no CUDA GPU visible to torch. Set CUDA_VISIBLE_DEVICES and" >&2
    echo "         check the driver (nvidia-smi). Override with SKIP_PREFLIGHT=1." >&2
    ok=0
  else
    echo "  [ok]   CUDA GPU visible"
  fi
  # 2. Data cache dir writable.
  if ! mkdir -p "$DUO_DATA_DIR" 2>/dev/null || [[ ! -w "$DUO_DATA_DIR" ]]; then
    echo "  [FAIL] DUO_DATA_DIR not writable: $DUO_DATA_DIR" >&2
    ok=0
  else
    echo "  [ok]   data cache dir writable ($DUO_DATA_DIR)"
  fi
  # 3. Only OpenWebText is guarded. Its per-split tokenized caches are
  #    <cache>/openwebtext-train_*.dat and openwebtext-valid_*.dat; when absent,
  #    get_dataset falls through to a raw load_dataset('openwebtext') build that
  #    downloads and extracts tens of GB and is fragile (DatasetGenerationError).
  #    Refuse it silently. Small corpora (wikitext103, ptb, ag_news, ...) build
  #    reliably in seconds, so they are not guarded.
  case "$DATA" in
    openwebtext*)
      if ! compgen -G "$DUO_DATA_DIR/openwebtext-*_*.dat" >/dev/null; then
        if [[ "${ALLOW_OWT_BUILD:-0}" == "1" ]]; then
          echo "  [warn] no tokenized OWT cache in $DUO_DATA_DIR; building from raw" \
               "(download + extract tens of GB, slow and fragile)."
        else
          echo "  [FAIL] no tokenized OpenWebText cache in $DUO_DATA_DIR" >&2
          echo "         (expected openwebtext-{train,valid}_*.dat). The raw OWT" >&2
          echo "         build is slow, disk-heavy, and fragile. Options:" >&2
          echo "           - point DUO_DATA_DIR at an existing cache, or" >&2
          echo "           - DATA=wikitext103 for a fast, reliable ablation, or" >&2
          echo "           - ALLOW_OWT_BUILD=1 to build OWT anyway." >&2
          ok=0
        fi
      else
        echo "  [ok]   tokenized OWT cache present"
      fi
      ;;
    *)
      echo "  [ok]   data=$DATA (small corpus, builds on demand)"
      ;;
  esac
  [[ "$ok" == "1" ]] || { echo "[preflight] aborting." >&2; exit 1; }
  echo "[preflight] all checks passed."
fi

for k in $KS; do
  echo "======================================================================"
  echo "[erlang] === training k=$k ($([ "$k" = 1 ] && echo 'time-conditioned MDLM baseline' || echo 'Erlang phases')) ==="
  echo "======================================================================"
  python -u -m main \
    algo=erlang \
    algo.erlang_k="$k" \
    model="$MODEL" \
    model.length="$LENGTH" \
    data="$DATA" \
    loader.batch_size="$BATCH_SIZE" \
    loader.eval_batch_size="$BATCH_SIZE" \
    sampling.predictor=ancestral_cache \
    eval.compute_generative_perplexity=True \
    eval.gen_ppl_step_budgets="[1,2,4,8,16]" \
    trainer.max_steps="$STEPS" \
    wandb.name="erlang-k${k}-${DATA}" \
    +wandb.offline="$WANDB_OFFLINE" \
    "$@"
done

echo "[erlang] ablation complete. Compare val/gen_ppl (and val/gen_ppl@{k}step)"
echo "[erlang] across runs: k=1 is the time-conditioned MDLM baseline, k>1 adds Erlang phases."
