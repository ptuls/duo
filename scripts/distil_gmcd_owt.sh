#!/bin/bash
#SBATCH -J gmcd                       # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

# G-MCD: Gumbel-lift consistency distillation of a pretrained MDLM.
# Unlike scripts/distil_owt.sh (Duo DCD), no integral cache and no
# curriculum flags are needed: the threshold coupling is exact for any
# mask schedule (Theorem 5.1 of the Gumbel-lift paper).

set -euo pipefail
export HYDRA_FULL_ERROR=1
finetune_path=${FINETUNE_PATH:-/path/to/mdlm.ckpt}

# ------------------------------------------------------------------ preflight
# Fail fast with a clear message rather than a Python traceback minutes in.
# Skip with SKIP_PREFLIGHT=1.
preflight() {
  local ok=1
  echo "[preflight] checking launch prerequisites..."

  # 1. Teacher checkpoint exists and is non-trivial (not the placeholder).
  if [[ "$finetune_path" == /path/to/* ]]; then
    echo "  [FAIL] FINETUNE_PATH is the placeholder ($finetune_path)."
    echo "         Set FINETUNE_PATH=/abs/path/to/mdlm.ckpt"; ok=0
  elif [[ ! -f "$finetune_path" ]]; then
    echo "  [FAIL] checkpoint not found: $finetune_path"; ok=0
  else
    local sz; sz=$(stat -c%s "$finetune_path" 2>/dev/null || stat -f%z "$finetune_path")
    if [[ "${sz:-0}" -lt 10000000 ]]; then
      echo "  [FAIL] checkpoint suspiciously small (${sz} bytes): $finetune_path"
      echo "         A partial gdown download? Re-fetch mdlm.ckpt."; ok=0
    else
      echo "  [ok]   checkpoint present ($((sz/1024/1024)) MB)"
    fi
  fi

  # 2. Data cache dir is writable (the old /share path was not).
  local dd="${DUO_DATA_DIR:-$PWD/data}"
  if ! mkdir -p "$dd" 2>/dev/null || [[ ! -w "$dd" ]]; then
    echo "  [FAIL] data cache dir not writable: $dd"
    echo "         Set DUO_DATA_DIR to writable storage."; ok=0
  else
    echo "  [ok]   data cache dir writable ($dd)"
  fi

  # 3. GPU visible.
  if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  [FAIL] torch.cuda.is_available() is False (no GPU / bad CUDA build)."; ok=0
  else
    local ng; ng=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "  [ok]   CUDA available, ${ng} GPU(s) visible"
  fi

  # 4. wandb reachable unless offline (online run needs a login/API key).
  if [[ "${WANDB_OFFLINE:-false}" != "true" ]]; then
    if ! python -c "import netrc,os,sys; sys.exit(0 if (os.getenv('WANDB_API_KEY') or ('api.wandb.ai' in (netrc.netrc().hosts if os.path.exists(os.path.expanduser('~/.netrc')) else {}))) else 1)" 2>/dev/null; then
      echo "  [FAIL] online W&B requested but not logged in."
      echo "         Run 'wandb login', or set WANDB_API_KEY, or WANDB_OFFLINE=true."; ok=0
    else
      echo "  [ok]   W&B credentials found (online logging)"
    fi
  else
    echo "  [ok]   W&B offline (local logging)"
  fi

  # 5. Warn (not fail) if the local branch is behind its remote.
  if git -C "$(dirname "$0")/.." rev-parse @ >/dev/null 2>&1; then
    git -C "$(dirname "$0")/.." fetch --quiet origin 2>/dev/null || true
    local behind; behind=$(git -C "$(dirname "$0")/.." rev-list --count @..@{u} 2>/dev/null || echo 0)
    if [[ "${behind:-0}" -gt 0 ]]; then
      echo "  [warn] local branch is $behind commit(s) behind origin. 'git pull' to get latest fixes."
    fi
  fi

  if [[ "$ok" -ne 1 ]]; then
    echo "[preflight] FAILED. Fix the above or re-run with SKIP_PREFLIGHT=1 to bypass." >&2
    exit 1
  fi
  echo "[preflight] all checks passed."
}
[[ "${SKIP_PREFLIGHT:-0}" == "1" ]] || preflight

# No srun/torchrun needed: trainer.devices=${device_count:} auto-detects all
# GPUs and the default ddp strategy spawns its own workers from one process.
# Set CUDA_VISIBLE_DEVICES to restrict which GPUs are used.
python -u -m main \
  mode=train \
  data.cache_dir=${DUO_DATA_DIR:-$PWD/data} \
  loader.batch_size=${BATCH_SIZE:-16} \
  loader.eval_batch_size=${EVAL_BATCH_SIZE:-32} \
  data=openwebtext-split \
  model=small \
  algo=gmcd \
  training.finetune_path=$finetune_path \
  sampling.num_sample_batches=10 \
  sampling.steps=32 \
  sampling.predictor=ancestral_cache \
  eval.compute_generative_perplexity=True \
  'eval.gen_ppl_step_budgets=[1,2,4,8]' \
  algo.T=512 \
  lr_scheduler.num_warmup_steps=500 \
  trainer.val_check_interval=1000 \
  trainer.max_steps=50000 \
  loader.global_batch_size=${GLOBAL_BATCH_SIZE:-128} \
  training.ema=0.999 \
  algo.update_teacher_every=10000 \
  optim.lr=6e-5 \
  trainer.limit_val_batches=8 \
  algo.teacher_ema=False \
  algo.linear_growth_dt=false \
  +wandb.offline=${WANDB_OFFLINE:-false} \
  "$@"   # extra overrides win, e.g. wandb.entity=you wandb.name=gmcd-run1 trainer.max_steps=50
