# Gumbel-lift additions to this Duo fork

Additions for the Gumbel-lift paper (`~/work/diffusion-algorithimic-information-theory/diffusion_duality/paper`).

## What was added

- `algo.GMCD` (`configs/algo/gmcd.yaml`, `algo=gmcd`): Gumbel-lift consistency
  distillation of a pretrained MDLM. Exact threshold coupling (one uniform per
  token vs both noise levels) instead of a Gaussian latent or F_Y calibration.
  Forward KL on tokens masked at t; SUBS carry-over supplies the
  committed-token cross-entropy automatically. Teacher sharpening is a config
  knob (`teacher_temp_*`), default off; Latent Shadows hardcodes 0.96->0.75.
- Dual-geometry sampling grids (`sampling.grid` in `configs/config.yaml`):
  `uniform_t` (default, byte-identical to before), `uniform_beta`,
  `uniform_logodds` (= geometric annealing), `uniform_hazard`. Implemented in
  `trainer_base._grid_time_profile` via `get_t_for_alpha`. Note: under
  LogLinear (beta_t = t), uniform_beta == uniform_t; logodds and hazard differ.
- `scripts/distil_gmcd_owt.sh`: full G-MCD launch (mirror of distil_owt.sh).
- `scripts/run_first_session.sh`: 1-GPU de-risking session (checkpoint check
  -> teacher baseline -> E3 grids). Run this FIRST.
- `scripts/smoke_gmcd.py`: CPU smoke test of GMCD (passing).
- `tests/test_gmcd.py`: regression suite (checkpoint round-trip, teacher
  determinism, kl-bwd finiteness, coupling, dt boundaries). `pytest tests/test_gmcd.py`.

## Review fixes (post code-review)

- Checkpoints strip `teacher.*` keys (`on_save/on_load_checkpoint`) so they
  reload and resume (teacher is rebuilt lazily on the first step).
- Teacher forced to `eval()` on every call: it is a registered submodule, so
  `model.train()` would otherwise re-enable dropout and add noise to targets.
- `kl-bwd` uses forward cross-entropy on tokens revealed in (s, t] (where SUBS
  makes the teacher a delta and reverse KL is infinite); the chosen KL applies
  only on the still-masked intersection. `kl-fwd` is unchanged.

## Known accepted risks

- `trainer_base.py` uses `trust_remote_code=True` for HF backbones and loads
  pickle caches (Duo integral cache). Standard research conveniences; only run
  with trusted checkpoints and data.

## Portability patches (for non-cluster boxes)

- `configs/config.yaml`: `num_workers` resolver falls back to `cpu_count()`
  when `os.sched_getaffinity` is absent (macOS).
- `dataloader.py`: guard `os.sched_getaffinity` (absent on macOS).
- `models/dit.py`: pure-torch rotary + SDPA fallback when `flash_attn` is
  unavailable. On A100s install flash-attn and this path is unused.

## First session

    TEACHER=/path/to/mdlm.ckpt bash scripts/run_first_session.sh

Then the 1-GPU G-MCD smoke, then the 8-GPU ablation
(loss_type in {kl-fwd,kl-bwd} x teacher_temp_start in {1.0,0.96}).

## E1: exact categorical semantics beyond one-hot states

`semantic_stress_test.py` compares direct categorical sampling, Gumbel noisy
argmax, and Gaussian noisy argmax against `softmax(logits / temperature)`. The
Gaussian scale is recalibrated for every vocabulary size and temperature to
match the canonical one-hot state, so the one-hot stratum is a fairness check
and the soft-logit strata test the actual arbitrary-logit claim.

Start with the CPU plumbing check:

    python semantic_stress_test.py --smoke --output-dir outputs/e1_smoke

Then run the controlled synthetic experiment (GPU recommended):

    python semantic_stress_test.py \
      --vocab-sizes 2 16 128 --temperatures 0.5 1 2 \
      --logit-scales 1 --interpolation-weights 0.5 --guidance-scales 2 \
      --n-vectors 32 --n-samples 32768 \
      --output-dir outputs/e1_synthetic

For real denoiser states, either pass a NumPy array with shape `(..., vocab)`
or collect logits directly from a checkpoint and local validation text:

    python semantic_stress_test.py \
      --real-only --real-logits-npy outputs/mdlm_logits.npy --real-top-k 512 \
      --n-vectors 32 --n-samples 32768 \
      --output-dir outputs/e1_real

    python semantic_stress_test.py \
      --real-only \
      --checkpoint /path/to/mdlm.ckpt --text-file /path/to/validation.txt \
      --real-top-k 512 --n-vectors 32 --n-samples 32768 \
      --output-dir outputs/e1_checkpoint

The report records the retained target mass for every top-k real-logit vector;
do not interpret a heavily truncated run as a full-vocabulary result. The key
comparison is excess error over the direct categorical Monte Carlo floor:
Gumbel should remain at the floor off the calibration state, while calibrated
Gaussian should not.

## E3: curvature-aware decoding (level-two flagship)

`curvature_decode.py` (+ `scripts/run_curvature_decode.sh`). MaskGIT-style
iterative unmasking where the commit order is set by the exact conditional
noise-score covariance term in the second-order Gumbel-Tweedie identity. For
the categorical clean-logit model `V = a e_X + b G`, the residual moments are
closed form in the denoiser posterior, so the estimator is O(V), uses one model
call per decoding step, and needs neither finite differences nor Hutchinson
probes. Orders compared by default: confidence (standard), entropy
(first-order control), exact curvature, random. The former continuous-input
finite-difference statistic remains available as the explicitly named
`sensitivity` ablation. Metrics: gen_ppl + distinct-2 at steps {4,8,16,32}.

Important interpretation: the moment calculation is exact given `p(X | V)`,
but a pretrained discrete MDLM supplies `p_theta(X | z)` and does not expose
the continuous Gumbel observation. The decoder uses that distribution as a
plug-in posterior on the neutral/equal-logit slice. On this slice, Tweedie
covariance trace is
`expm1(a / b)^2 / b^2 * (1 - sum(p^2))`, so its position ranking is exactly
Gini-impurity ranking. Thus a win tests Gini-based allocation rather than a
uniquely Gumbel-specific signal. End-to-end exact conditioning requires a
logit-space denoiser and the actual lifted observation; the code exposes the
residual-dependent injection, Hessian-vector product, and marginal-Hessian
diagonal for that setting.

Status: plumbing smoke-tested on CPU (toy model). **Not yet validated on a
1. Exact curvature: no probe calibration; `--signal-scale` and
   `--gumbel-scale` change only a common positive factor on the neutral slice.
2. Sensitivity ablation only: calibrate `--in-scale`, `--n-probes`, and
   `--eps`; too much input sharpness can collapse the statistic.
Cost: exact curvature matches confidence at one NFE per step. Sensitivity costs
~(n_probes+2)x the forwards of confidence per step.

## E3 mechanism gate: validate exact curvature first

`curvature_validate.py` (+ `scripts/run_curvature_validate.sh`). Before claiming
better decoding, establish that the exact curvature statistic predicts, on real
masked MDLM states, which position is safe to commit next. Targets: correctness
(p of true token), stability (does argmax-now survive resolving the rest of the
context to GT), stability probability, and oracle_gain
(remaining NLL after committing i, over a try-every-candidate oracle). Controls:
entropy, Gini, max_prob, logit margin, distance from uniform, random ranking,
the former sensitivity proxy, and shuffled nulls. Metrics: Spearman,
top-decile enrichment and lift, and position-selection regret.

**Gate:** the exact curvature/Gini ranking must beat entropy, margin, and its
shuffled null on stability before running the full decoding experiment. Its
algebraic equality with Gini on the neutral slice must be reported. Plumbing is
smoke-tested on CPU; a real signal requires the trained model on GPU.

## Token-level A_i characterization (resolves the char-vocab caveat)

`characterize_Ai_token.py` (+ `scripts/run_characterize_Ai_token.sh`). Runs the
A_i characterization on a REAL MDLM's contextual posteriors, on a 50k vocab.
Reports (1) effective plausible support (participation ratio, entropy,
tokens-for-90%-mass) -- the decisive number: if it's tens-hundreds, A_i =
(top-m, 3-bit) is a genuine compression; (2) the top-m/B retention of the
residual curvature within fixed-Gini bins, with commitment_gain from MDLM
leave-one-out. Plumbing smoke-tested on CPU (toy); real numbers need the trained
MDLM on GPU. This must pass before building the augmented-state model.

## Erlang-k semi-Markov masking (algo=erlang_mdlm)

`algo.ErlangMDLM` (`configs/algo/erlang.yaml`, `algo=erlang`). The real-model
test of the semi-Markov appendix ("Semi-Markov masking: age-dependent hazards
and Erlang shades"). Each token's reveal level advances through `k` stages
(total holding time Erlang-k), so the denoiser sees a graded mixture of the
true token and `[MASK]` rather than a hard binary mask -- "k shades of masking".
`k=1` is the memoryless exponential = plain MDLM.

- Forward: phase `j ~ Binomial(k, (1-alpha_t)^{1/k})`, so `P(fully masked)` still
  equals `1 - alpha_t`; reveal fraction `r = 1 - j/k`.
- Input: graded embedding via the backbone `embedding_bag` weighted path,
  index pair `[token, mask]` with weights `[r, 1-r]`. At the endpoints
  (`r in {0,1}`) this is byte-identical to MDLM's hard mask/token embedding, so
  the inherited MDLM ancestral sampler runs unchanged and gen_ppl stays
  comparable. Only intermediate shades enrich the training signal.
- Loss: reveal-weighted denoising CE = MDLM continuous weight
  `dalpha_t/(1-alpha_t)` times masked fraction `j/k`. Reduces exactly to the
  MDLM SUBS objective at `k=1`; for `k>1` it is an approximate (not exact-ELBO)
  objective, stated as such. The question it answers is empirical: does
  phase-graded training lower gen_ppl over hard masking?

`scripts/smoke_erlang.py` (CPU, passing): phase marginal, endpoint-embedding
equivalence, exact `k=1` reduction to MDLM on masked positions (max|diff|=0),
and finite backward for `k in {1,2,4}`.

`scripts/run_erlang_ablation.sh`: controlled OWT ablation over `k in {1,2,4}`
under identical settings, logging `val/gen_ppl` and few-step budgets. Since
`k=1` is the MDLM baseline, any gen_ppl gap at `k>1` isolates the shades effect.
This is the real-model counterpart of the analytic phase gate
(`experiments/gumbel_lift/semimarkov_phase_gate.py`), which found the phase
carries decision-relevant signal only for `k>1`.

Status: built and CPU-smoke-tested; the ablation needs the GPU.
