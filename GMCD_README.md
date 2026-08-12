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

## E3: curvature-aware decoding (level-two flagship)

`curvature_decode.py` (+ `scripts/run_curvature_decode.sh`). MaskGIT-style
iterative unmasking where the commit order is set by posterior curvature instead
of confidence. Curvature reads the continuous logit coordinate (the DiT's
soft-input path `softmax(l)@E`, x.ndim==3) via a Hutchinson estimate of the
output-vs-input Jacobian trace, so it cannot reduce to the first-order
prediction. Orders compared: confidence (standard), entropy (first-order
control), curvature (ours), random. Metrics: gen_ppl + distinct-2 at
steps {4,8,16,32}. The decisive contrast is curvature vs entropy.

Status: plumbing smoke-tested on CPU (toy model). **Not yet validated on a
trained model.** Two things to calibrate on GPU before trusting results:
1. `--in-scale`: too high saturates softmax and the sensitivity -> 0; check the
   curvature scores have spread across positions.
2. `--n-probes` / `--eps`: enough probes for a stable estimate.
Cost: curvature costs ~(n_probes+2)x the forwards of confidence per step.

## E3 mechanism gate: validate curvature is a real signal FIRST

`curvature_validate.py` (+ `scripts/run_curvature_validate.sh`). Before claiming
better decoding, establish that the curvature statistic predicts, on real masked
MDLM states, which position is safe to commit next. Targets: token correctness
(p of true token), stability (does argmax-now survive resolving the rest of the
context to GT -- the second-order property entropy cannot see), and oracle_gain
(remaining NLL after committing i, over a try-every-candidate oracle). Controls:
entropy, max_prob, logit margin, first-order score magnitude, random, and two
nulls (curvature_shuffled, curvature_diag). Metrics: Spearman, top-decile
enrichment, position-selection regret.

**Gate:** curvature must beat entropy AND margin AND its shuffled null on
stability. If not, do NOT run the decoding experiment -- a downstream win could
not be attributed to the theory. Run this before run_curvature_decode.sh.
Plumbing smoke-tested on CPU (toy); real signal needs the trained model on GPU.
