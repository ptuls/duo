"""E3: exact Gumbel-Tweedie curvature decoding for masked diffusion.

The claim under test is whether the exact posterior-score covariance term in
the second-order Gumbel-Tweedie identity improves parallel unmasking order over
confidence. For the categorical clean-logit model

    V = a e_X + b G,  G_i iid Gumbel(0, 1),

all conditional residual moments can be evaluated exactly from the denoiser's
categorical posterior. No finite differences or Hutchinson probes are needed.

Decoder. MaskGIT-style iterative unmasking in `k` steps. At each step the
denoiser predicts and samples a categorical token for every masked position;
we commit the most "decidable" positions and iterate. Sampling is essential:
an argmax fill would duplicate every deterministic sample and invalidate the
quality-diversity comparison. The methods differ only in the ranking score:

  confidence  commit highest max-prob first          (the standard rule)
  random      commit a random subset                 (control)
  entropy     commit lowest predictive entropy first  (first-order only)
  curvature   commit lowest exact Tweedie covariance first
  sensitivity commit lowest finite-difference sensitivity first (ablation)

Exact estimator. ``curvature`` is the trace of the conditional covariance of
the Gumbel noise score. The standalone moment calculation is exact given
``p(X | V)``. A pretrained discrete MDLM instead supplies ``p_theta(X | z)``;
the decoder uses it as a plug-in posterior on the neutral lifted-logit slice.
There, curvature is a fixed positive multiple of categorical Gini impurity
1 - sum_k p_k^2. The implementation also exposes the residual-dependent noise
injection and marginal-Hessian diagonal for non-neutral continuous logits.

The optional ``sensitivity`` ablation feeds continuous input logits through
the DiT soft-input path and estimates tr(J^T J) by finite differences. It is a
different statistic and must not be reported as Tweedie curvature.

Usage:
  # real model (GPU):
  python curvature_decode.py --checkpoint checkpoints/mdlm.ckpt \
      --orders confidence entropy curvature random \
      --steps 2 4 8 16 32 --n-samples 512 --judge gpt2-large
  # CPU plumbing smoke (random-weight tiny model):
  python curvature_decode.py --model toy --steps 4 8 --n-samples 8 --no-ppl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GumbelTweedieTerms:
    """Diagonal terms in the exact second-order Gumbel-Tweedie identity.

    All tensors have shape ``(..., vocabulary_size)``. ``noise_curvature_diag``
    is ``-E[J_N(R) | V]`` and is therefore the positive noise-injection term.
    The marginal log-density Hessian is
    ``-noise_curvature_diag + score_covariance_diag``.
    """

    residual_exponential_mean: torch.Tensor
    marginal_score: torch.Tensor
    noise_curvature_diag: torch.Tensor
    score_covariance_diag: torch.Tensor
    marginal_hessian_diag: torch.Tensor

    @property
    def noise_curvature_trace(self):
        """Returns the trace of the positive noise-curvature injection."""
        return self.noise_curvature_diag.sum(dim=-1)

    @property
    def score_covariance_trace(self):
        """Returns the trace of the posterior noise-score covariance."""
        return self.score_covariance_diag.sum(dim=-1)

    @property
    def marginal_hessian_trace(self):
        """Returns the trace of the marginal log-density Hessian."""
        return self.marginal_hessian_diag.sum(dim=-1)


# --------------------------------------------------------------------- model
def build_model(args):
    """Return a masked-diffusion model exposing .backbone, .forward, .mask_index,
    .vocab_size, .noise, ._process_sigma, ._sigma_from_alphat, .tokenizer."""
    import hydra
    import transformers
    import algo

    overrides = [
        "algo=mdlm",
        "data=openwebtext-split",
        "sampling.predictor=ancestral_cache",
        f'model={"tiny" if args.model == "toy" else getattr(args, "architecture", "small")}',
        "loader.batch_size=1",
        "loader.eval_batch_size=1",
        "loader.global_batch_size=1",
        "trainer.precision=32",
    ]
    with hydra.initialize(config_path="configs", version_base=None):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    if args.model == "toy":
        model = algo.MDLM(cfg, tokenizer=tok)
    else:
        model = algo.MDLM.load_from_checkpoint(
            args.checkpoint, tokenizer=tok, config=cfg
        )
    model = model.to(args.device).eval()
    if getattr(args, "disable_ema", False):
        model.ema = None
    elif model.ema is not None:
        # Lightning normally installs EMA weights in its validation hook. These
        # standalone experiments call model.forward directly, so do it explicitly.
        model.ema.move_shadow_params_to_device(args.device)
        model._eval_mode()
    return model


# ------------------------------------------------------------- scoring rules
@torch.no_grad()
def denoise_logp(model, z, sigma):
    """Per-position log-probabilities (B, L, V), mask column excluded."""
    return model.forward(z, sigma=sigma)


def confidence_score(logp):
    # higher max log-prob -> more decidable -> commit first
    return logp.float().max(dim=-1).values


def neg_entropy_score(logp):
    # -entropy; higher (less negative) -> lower entropy -> commit first
    logp = logp.float()
    p = logp.exp()
    safe_logp = torch.where(p > 0, logp, torch.zeros_like(logp))
    return (p * safe_logp).sum(dim=-1)


def sample_clean_tokens(logp, mask_index, generator):
    """Samples token IDs while assigning exactly zero mass to the mask token."""
    probabilities = logp.float().exp()
    probabilities[..., mask_index] = 0
    normalizer = probabilities.sum(dim=-1, keepdim=True)
    if bool((normalizer <= 0).any()):
        raise ValueError("denoiser assigned zero probability to every clean token")
    probabilities /= normalizer
    shape = probabilities.shape[:-1]
    return torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator
    ).reshape(shape)


@torch.no_grad()
def _gumbel_tweedie_inputs(
    logp,
    mask_index,
    signal_scale,
    noise_scale,
    observed_logits,
):
    if signal_scale <= 0 or not math.isfinite(signal_scale):
        raise ValueError("signal_scale must be finite and positive")
    if noise_scale <= 0 or not math.isfinite(noise_scale):
        raise ValueError("noise_scale must be finite and positive")
    if logp.ndim < 1 or logp.shape[-1] < 2:
        raise ValueError("logp must have a vocabulary dimension of size at least two")

    posterior_logp = logp.detach().float().clone()
    if mask_index is not None:
        if not -posterior_logp.shape[-1] <= mask_index < posterior_logp.shape[-1]:
            raise ValueError("mask_index is outside the vocabulary")
        posterior_logp[..., mask_index] = -math.inf
    log_normalizer = torch.logsumexp(posterior_logp, dim=-1, keepdim=True)
    if not bool(torch.isfinite(log_normalizer).all()):
        raise ValueError("denoiser assigned zero probability to every clean token")
    posterior = torch.exp(posterior_logp - log_normalizer)

    if observed_logits is None:
        inverse_residual_base = torch.ones_like(posterior)
    else:
        if observed_logits.shape != logp.shape:
            raise ValueError("observed_logits must have the same shape as logp")
        inverse_residual_base = torch.exp(
            -observed_logits.detach().float() / noise_scale
        )
        if not bool(torch.isfinite(inverse_residual_base).all()):
            raise ValueError("observed_logits produce non-finite residual moments")

    signal_multiplier = math.expm1(signal_scale / noise_scale)
    if not math.isfinite(signal_multiplier * signal_multiplier):
        raise ValueError("signal_scale / noise_scale is too large for float arithmetic")
    return posterior, inverse_residual_base, signal_multiplier


@torch.no_grad()
def gumbel_tweedie_terms(
    logp,
    mask_index=None,
    signal_scale=1.0,
    noise_scale=1.0,
    observed_logits=None,
):
    """Evaluates the exact categorical Gumbel-Tweedie terms in O(V) memory.

    The clean latent is ``U=e_X`` and the lifted observation is
    ``V=signal_scale * U + noise_scale * G``. Given the posterior
    ``p(X | V)`` in ``logp``, this computes the diagonal of

      Hess log p(V) = E[J_N(V-aU) | V] + Cov(s_N(V-aU) | V).

    ``observed_logits=None`` selects the neutral slice ``V=0``. On that slice,
    the score-covariance trace is
    ``expm1(a/b)^2 / b^2 * (1 - ||p||_2^2)``. A discrete MDLM does not expose
    its underlying continuous Gumbel draw, so using its discrete posterior here
    is an explicit plug-in assumption. End-to-end exact conditioning requires
    a logit-space denoiser and the actual lifted observation.

    Args:
        logp: Model log posterior with shape ``(..., vocabulary_size)``.
        mask_index: Optional class to exclude from the clean latent support.
        signal_scale: Positive clean one-hot amplitude ``a``.
        noise_scale: Positive Gumbel scale ``b``.
        observed_logits: Optional lifted observation ``V``, shaped as ``logp``.

    Returns:
        Exact first- and second-order diagonal terms in float32.

    Raises:
        ValueError: If parameters, shapes, or the clean posterior are invalid.
    """
    posterior, inverse_residual_base, signal_multiplier = _gumbel_tweedie_inputs(
        logp,
        mask_index,
        signal_scale,
        noise_scale,
        observed_logits,
    )

    residual_exponential_mean = inverse_residual_base * (
        1.0 + signal_multiplier * posterior
    )
    inverse_noise_variance = 1.0 / (noise_scale * noise_scale)
    noise_curvature_diag = residual_exponential_mean * inverse_noise_variance
    score_covariance_diag = (
        signal_multiplier
        * signal_multiplier
        * inverse_noise_variance
        * inverse_residual_base.square()
        * posterior
        * (1.0 - posterior)
    )
    marginal_hessian_diag = score_covariance_diag - noise_curvature_diag
    marginal_score = (residual_exponential_mean - 1.0) / noise_scale
    return GumbelTweedieTerms(
        residual_exponential_mean=residual_exponential_mean,
        marginal_score=marginal_score,
        noise_curvature_diag=noise_curvature_diag,
        score_covariance_diag=score_covariance_diag,
        marginal_hessian_diag=marginal_hessian_diag,
    )


@torch.no_grad()
def gumbel_tweedie_hessian_vector_product(
    logp,
    vector,
    mask_index=None,
    signal_scale=1.0,
    noise_scale=1.0,
    observed_logits=None,
):
    """Applies the complete exact marginal Hessian without materializing V by V.

    Args:
        logp: Model log posterior with shape ``(..., vocabulary_size)``.
        vector: Vector to multiply, with the same shape as ``logp``.
        mask_index: Optional class to exclude from the clean latent support.
        signal_scale: Positive clean one-hot amplitude ``a``.
        noise_scale: Positive Gumbel scale ``b``.
        observed_logits: Optional lifted observation ``V``, shaped as ``logp``.

    Returns:
        ``(Hess log p(V)) @ vector`` in float32 and O(V) memory.

    Raises:
        ValueError: If ``vector`` does not have the same shape as ``logp``.
    """
    if vector.shape != logp.shape:
        raise ValueError("vector must have the same shape as logp")
    posterior, inverse_residual_base, signal_multiplier = _gumbel_tweedie_inputs(
        logp,
        mask_index,
        signal_scale,
        noise_scale,
        observed_logits,
    )
    vector = vector.detach().float()
    weighted_vector = inverse_residual_base * vector
    posterior_projection = (posterior * weighted_vector).sum(dim=-1, keepdim=True)
    inverse_noise_variance = 1.0 / (noise_scale * noise_scale)
    covariance_product = (
        signal_multiplier
        * signal_multiplier
        * inverse_noise_variance
        * inverse_residual_base
        * posterior
        * (weighted_vector - posterior_projection)
    )
    residual_exponential_mean = inverse_residual_base * (
        1.0 + signal_multiplier * posterior
    )
    return covariance_product - (
        inverse_noise_variance * residual_exponential_mean * vector
    )


@torch.no_grad()
def curvature_score(logp, mask_index, signal_scale=1.0, noise_scale=1.0):
    """Returns negative exact posterior curvature for commit-first ranking.

    Higher scores mean lower posterior noise-score covariance and therefore a
    more decisive position. The neutral lifted-logit slice is intentional; see
    ``gumbel_tweedie_terms`` for its interpretation.
    """
    terms = gumbel_tweedie_terms(
        logp,
        mask_index=mask_index,
        signal_scale=signal_scale,
        noise_scale=noise_scale,
    )
    return -terms.score_covariance_trace


@torch.no_grad()
def sensitivity_score(model, z, sigma, masked, n_probes, eps, in_scale, generator=None):
    """Negative Hutchinson sensitivity proxy, so commit-first == low sensitivity.

    Feeds continuous input logits (soft-input embedding path), perturbs the masked
    context, and averages the squared change in each position's predicted
    distribution. This is tr(J^T J) of the output-vs-input Jacobian restricted to
    masked inputs, an unbiased Hutchinson estimate up to the finite-difference eps.
    """
    V = model.vocab_size
    base = in_scale * F.one_hot(z, V).to(torch.float32)  # (B, L, V), near one-hot
    with torch.amp.autocast("cuda", enabled=False):
        # Use TrainerBase.forward's continuous-input hook so MDLM's SUBS output
        # processing still removes the impossible mask class. Calling the backbone
        # directly here used to measure a different, invalid categorical law.
        p0 = model.forward(z, sigma=sigma, nn_input_idxs=base).float().exp()
        sens = torch.zeros(z.shape, device=z.device, dtype=torch.float32)
        m = masked.unsqueeze(-1).to(base.dtype)  # (B, L, 1) perturb masked only
        for _ in range(n_probes):
            noise = torch.randn(
                base.shape, dtype=base.dtype, device=base.device, generator=generator
            )
            pert = base + eps * noise * m
            p = model.forward(z, sigma=sigma, nn_input_idxs=pert).float().exp()
            sens += ((p - p0) ** 2).sum(dim=-1) / (eps * eps)
    return -(sens / n_probes)


# -------------------------------------------------------------------- decoder
def unmask_schedule(L, n_steps):
    """Cosine schedule: number of positions to reveal after each step (sums to L)."""
    # fraction still masked after step i follows cos; reveal the difference.
    masked_frac = np.cos(np.linspace(0, 1, n_steps + 1) * math.pi / 2)
    masked_frac[0], masked_frac[-1] = 1.0, 0.0
    counts = np.round(masked_frac * L).astype(int)
    reveal = -np.diff(counts)  # positions revealed at each of n_steps steps
    reveal[-1] += L - reveal.sum()  # pin total to L
    return reveal.tolist()


@torch.no_grad()
def decode(model, order, n_steps, n_samples, L, args, token_gen, score_gen, probe_gen):
    device = args.device
    mask_id, V = model.mask_index, model.vocab_size
    out, nfe = [], 0
    t0 = time.perf_counter()
    reveal = unmask_schedule(L, n_steps)
    for start in range(0, n_samples, args.batch_size):
        b = min(args.batch_size, n_samples - start)
        z = torch.full((b, L), mask_id, dtype=torch.long, device=device)
        for step in range(n_steps):
            masked = z == mask_id
            n_masked = int(masked[0].sum())
            if n_masked == 0:
                break
            # sigma from the current masked fraction (alpha = 1 - frac_masked)
            frac = masked.float().mean(dim=1, keepdim=True).clamp(1e-4, 1 - 1e-4)
            sigma = model._sigma_from_alphat(1 - frac)
            logp = denoise_logp(model, z, sigma)
            nfe += 1
            if order == "confidence":
                score = confidence_score(logp)
            elif order == "entropy":
                score = neg_entropy_score(logp)
            elif order == "curvature":
                score = curvature_score(
                    logp,
                    mask_id,
                    signal_scale=args.signal_scale,
                    noise_scale=args.gumbel_scale,
                )
            elif order == "sensitivity":
                score = sensitivity_score(
                    model,
                    z,
                    sigma,
                    masked,
                    args.n_probes,
                    args.eps,
                    args.in_scale,
                    probe_gen,
                )
                nfe += args.n_probes + 1
            elif order == "random":
                score = torch.rand(z.shape, device=device, generator=score_gen)
            else:
                raise ValueError(order)
            score = score.masked_fill(~masked, -math.inf)  # only rank masked
            k = min(int(reveal[step]), n_masked)
            if k <= 0:
                continue
            idx = score.topk(k, dim=1).indices  # positions to commit
            fill = sample_clean_tokens(logp, mask_id, token_gen)
            z = z.scatter(1, idx, fill.gather(1, idx))
        # fill any leftover masked positions
        left = z == mask_id
        if left.any():
            logp = denoise_logp(
                model,
                z,
                model._sigma_from_alphat(
                    torch.full((z.shape[0], 1), 1e-3, device=device)
                ),
            )
            nfe += 1
            z = torch.where(left, sample_clean_tokens(logp, mask_id, token_gen), z)
        out.append(z.cpu())
    return torch.cat(out), nfe, time.perf_counter() - t0


# ------------------------------------------------------------------ metrics
def distinct_n(tokens, n):
    grams, total = set(), 0
    for row in tokens.tolist():
        for i in range(len(row) - n + 1):
            grams.add(tuple(row[i : i + n]))
            total += 1
    return len(grams) / max(total, 1)


@torch.no_grad()
def gen_ppl(tokens, judge_name, device, batch_size=8):
    from transformers import AutoModelForCausalLM

    judge = AutoModelForCausalLM.from_pretrained(judge_name).to(device).eval()
    nll, count = 0.0, 0
    for i in range(0, tokens.shape[0], batch_size):
        ids = tokens[i : i + batch_size].to(device)
        nll += float(judge(ids, labels=ids).loss) * (ids.numel() - ids.shape[0])
        count += ids.numel() - ids.shape[0]
    return float(np.exp(nll / max(count, 1)))


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["toy", "ckpt"], default="ckpt")
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--architecture",
        choices=["tiny", "small", "medium"],
        default="small",
        help="architecture used by the checkpoint",
    )
    p.add_argument(
        "--disable-ema",
        action="store_true",
        help="evaluate raw checkpoint weights instead of EMA weights",
    )
    p.add_argument(
        "--orders", nargs="+", default=["confidence", "entropy", "curvature", "random"]
    )
    p.add_argument("--steps", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument(
        "--signal-scale",
        type=float,
        default=1.0,
        help="clean one-hot amplitude a in the exact Tweedie model",
    )
    p.add_argument(
        "--gumbel-scale",
        type=float,
        default=1.0,
        help="Gumbel noise scale b in the exact Tweedie model",
    )
    p.add_argument(
        "--n-probes", type=int, default=4, help="sensitivity-ablation probes"
    )
    p.add_argument(
        "--eps", type=float, default=0.1, help="sensitivity finite-difference step"
    )
    p.add_argument(
        "--in-scale",
        type=float,
        default=6.0,
        help="sensitivity-ablation input sharpness. MUST be calibrated on GPU: too "
        "high saturates softmax and the sensitivity collapses to "
        "~0, too low blurs the state. Check that curvature scores "
        "have spread across positions (not all near-equal).",
    )
    p.add_argument("--judge", default="gpt2-large")
    p.add_argument("--no-ppl", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="curvature_decode_results.json")
    args = p.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "ckpt" and not args.checkpoint:
        p.error("--checkpoint required unless --model toy")
    if args.model == "toy":
        args.seq_len = 32
    if any(step <= 0 for step in args.steps):
        p.error("--steps values must be positive")
    if args.n_samples <= 0 or args.batch_size <= 0 or args.seq_len <= 0:
        p.error("sample, batch, and sequence sizes must be positive")
    if args.signal_scale <= 0 or args.gumbel_scale <= 0:
        p.error("--signal-scale and --gumbel-scale must be positive")
    if args.n_probes <= 0 or args.eps <= 0 or args.in_scale <= 0:
        p.error("--n-probes, --eps, and --in-scale must be positive")
    unknown_orders = set(args.orders) - {
        "confidence",
        "entropy",
        "curvature",
        "sensitivity",
        "random",
    }
    if unknown_orders:
        p.error(f"unknown orders: {sorted(unknown_orders)}")

    torch.manual_seed(args.seed)
    model = build_model(args)
    model_length = getattr(getattr(model.config, "model", None), "length", args.seq_len)
    L = min(args.seq_len, model_length)
    if "curvature" in args.orders:
        print(
            "[e3] curvature = exact neutral-lift Gumbel-Tweedie covariance "
            "with the MDLM posterior as a plug-in; its ranking equals Gini.",
            flush=True,
        )

    results = []
    for order in args.orders:
        for n_steps in args.steps:
            # Reinitialize independent streams for every method. Token noise is
            # paired across methods at a fixed step budget; score/probe randomness
            # cannot shift the token stream and confound the comparison.
            token_gen = torch.Generator(device=args.device).manual_seed(
                args.seed + 10_000 * n_steps + 1
            )
            score_gen = torch.Generator(device=args.device).manual_seed(
                args.seed + 10_000 * n_steps + 2
            )
            probe_gen = torch.Generator(device=args.device).manual_seed(
                args.seed + 10_000 * n_steps + 3
            )
            tokens, nfe, secs = decode(
                model,
                order,
                n_steps,
                args.n_samples,
                L,
                args,
                token_gen,
                score_gen,
                probe_gen,
            )
            row = {
                "order": order,
                "n_steps": n_steps,
                "nfe_per_pass": nfe
                / max(1, math.ceil(args.n_samples / args.batch_size)),
                "distinct2": distinct_n(tokens, 2),
                "wall_seconds": secs,
            }
            if not args.no_ppl:
                row["gen_ppl"] = gen_ppl(tokens, args.judge, args.device)
            results.append(row)
            print(f"[e3] {row}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[e3] wrote {args.output}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
