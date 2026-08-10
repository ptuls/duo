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

## Portability patches (for non-cluster boxes)

- `dataloader.py`: guard `os.sched_getaffinity` (absent on macOS).
- `models/dit.py`: pure-torch rotary + SDPA fallback when `flash_attn` is
  unavailable. On A100s install flash-attn and this path is unused.

## First session

    TEACHER=/path/to/mdlm.ckpt bash scripts/run_first_session.sh

Then the 1-GPU G-MCD smoke, then the 8-GPU ablation
(loss_type in {kl-fwd,kl-bwd} x teacher_temp_start in {1.0,0.96}).
