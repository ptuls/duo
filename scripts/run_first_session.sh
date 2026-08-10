#!/usr/bin/env bash
# First GPU session for the Gumbel-lift paper (8xA100 box).
#
# Chains the three cheap, de-risking steps before the expensive distillation
# run. All three run on ONE GPU and finish in a few hours. The point is to
# validate the checkpoint, produce the teacher baseline you must beat, and get
# one complete result (E3 grids) that needs no training -- BEFORE committing
# GPU-days to E2.
#
# Usage:
#   TEACHER=/path/to/mdlm.ckpt bash scripts/run_first_session.sh
#
# Env overrides:
#   TEACHER          (required) path to the pretrained MDLM Lightning .ckpt
#   DATA             dataset config name         (default openwebtext-split)
#   MODEL            model config name           (default small)
#   STEPS_GRID       few-step budgets to sweep   (default "1 2 4 8 16 32")
#   GRID_STEPS       budgets for the E3 grids    (default "2 4 8")
#   EVAL_BATCH       per-GPU eval batch size     (default 16)
#   SAMPLE_BATCHES   sample batches for gen-ppl  (default 4)
#   OUT             results dir                  (default watch_folder/first_session)

set -euo pipefail
cd "$(dirname "$0")/.."

: "${TEACHER:?set TEACHER=/path/to/mdlm.ckpt}"
DATA="${DATA:-openwebtext-split}"
MODEL="${MODEL:-small}"
STEPS_GRID="${STEPS_GRID:-1 2 4 8 16 32}"
GRID_STEPS="${GRID_STEPS:-2 4 8}"
EVAL_BATCH="${EVAL_BATCH:-16}"
SAMPLE_BATCHES="${SAMPLE_BATCHES:-4}"
OUT="${OUT:-watch_folder/first_session}"

export HYDRA_FULL_ERROR=1
mkdir -p "$OUT"

common=(data="$DATA" model="$MODEL" algo=mdlm
        eval.checkpoint_path="$TEACHER"
        loader.eval_batch_size="$EVAL_BATCH"
        +wandb.offline=true)

echo "=============================================================="
echo "STEP 0  checkpoint loads + perplexity eval (blocks everything)"
echo "=============================================================="
# If this fails, the checkpoint format does not match the fork's loader.
# Stop and fix before anything else (may need a HF->Lightning conversion or
# algo.backbone=hf_dit).
python -u -m main mode=ppl_eval \
  "${common[@]}" \
  sampling.num_sample_batches=0 \
  2>&1 | tee "$OUT/step0_ppl_eval.log"

echo "=============================================================="
echo "STEP 1  teacher baseline: gen-ppl at each step budget"
echo "=============================================================="
# The curve every distilled student must beat. Uniform-t grid (convention).
for s in $STEPS_GRID; do
  echo "--- teacher, $s steps ---"
  python -u -m main mode=sample_eval \
    "${common[@]}" \
    sampling.predictor=ancestral_cache \
    sampling.steps="$s" \
    sampling.grid=uniform_t \
    sampling.num_sample_batches="$SAMPLE_BATCHES" \
    eval.compute_generative_perplexity=True \
    2>&1 | tee "$OUT/step1_teacher_steps${s}.log"
done

echo "=============================================================="
echo "STEP 2  E3: dual-geometry sampling grids (teacher only)"
echo "=============================================================="
# Claim: grids uniform in the dual geometry beat uniform-t at small step
# counts. Under LogLinear, uniform_beta == uniform_t, so the informative
# grids are logodds (= geometric annealing) and hazard. Compared against the
# uniform_t numbers from STEP 1 at the same budgets.
for s in $GRID_STEPS; do
  for grid in uniform_logodds uniform_hazard; do
    echo "--- teacher, $s steps, grid=$grid ---"
    python -u -m main mode=sample_eval \
      "${common[@]}" \
      sampling.predictor=ancestral_cache \
      sampling.steps="$s" \
      sampling.grid="$grid" \
      sampling.num_sample_batches="$SAMPLE_BATCHES" \
      eval.compute_generative_perplexity=True \
      2>&1 | tee "$OUT/step2_${grid}_steps${s}.log"
  done
done

echo "=============================================================="
echo "DONE. Logs in $OUT/"
echo "Next: 1-GPU G-MCD smoke (scripts/distil_gmcd_owt.sh with"
echo "trainer.max_steps=500), then the full 8-GPU run."
echo "Grep gen-ppl:  grep -ri 'gen.*perplexity\\|gen_ppl' $OUT/"
echo "=============================================================="
