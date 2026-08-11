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

export HYDRA_FULL_ERROR=1
finetune_path=${FINETUNE_PATH:-/path/to/mdlm.ckpt}

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
  +wandb.offline=true
