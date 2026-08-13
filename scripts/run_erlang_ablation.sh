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
export DUO_DATA_DIR=${DUO_DATA_DIR:-$PWD/data}
mkdir -p "$DUO_DATA_DIR" watch_folder

echo "[erlang] ablation over k in: $KS"
echo "[erlang] steps=$STEPS batch=$BATCH_SIZE model=$MODEL length=$LENGTH"
echo "[erlang] data cache: $DUO_DATA_DIR"

for k in $KS; do
  echo "======================================================================"
  echo "[erlang] === training k=$k ($([ "$k" = 1 ] && echo 'time-conditioned MDLM baseline' || echo 'Erlang phases')) ==="
  echo "======================================================================"
  python -u -m main \
    algo=erlang \
    algo.erlang_k="$k" \
    model="$MODEL" \
    model.length="$LENGTH" \
    data=openwebtext-split \
    loader.batch_size="$BATCH_SIZE" \
    loader.eval_batch_size="$BATCH_SIZE" \
    sampling.predictor=ancestral_cache \
    eval.compute_generative_perplexity=True \
    eval.gen_ppl_step_budgets="[1,2,4,8,16]" \
    trainer.max_steps="$STEPS" \
    wandb.name="erlang-k${k}-owt" \
    +wandb.offline=True \
    "$@"
done

echo "[erlang] ablation complete. Compare val/gen_ppl (and val/gen_ppl@{k}step)"
echo "[erlang] across runs: k=1 is the time-conditioned MDLM baseline, k>1 adds Erlang phases."
