r"""E1: stress-test exact softmax semantics away from one-hot states.

The canonicality theorem says that adding i.i.d. standard Gumbel noise to
scaled logits and taking argmax samples ``softmax(logits / temperature)`` for
*every* logit vector. A Gaussian lift can be calibrated to agree on a chosen
one-hot reference state, but it should not preserve the same categorical law
on arbitrary soft logits.

This experiment compares three samplers:

* ``categorical``: samples the target distribution directly and measures the
  finite-Monte-Carlo error floor;
* ``gumbel``: argmax(logits / temperature + standard Gumbel noise);
* ``gaussian``: argmax(logits / temperature + sigma * standard Gaussian
  noise), where sigma is calibrated to match the target winner probability on
  the one-hot reference state for every vocabulary size and temperature.

Synthetic regimes include the calibration state, random soft logits,
interpolants between categorical corners, and classifier-free-guidance-style
extrapolations. Real denoiser logits can be read from NumPy or collected from
an MDLM checkpoint.

Examples:

  python semantic_stress_test.py --smoke

  python semantic_stress_test.py \
      --vocab-sizes 2 16 128 --temperatures 0.5 1 2 \
      --logit-scales 1 --interpolation-weights 0.5 --guidance-scales 2 \
      --n-vectors 32 --n-samples 32768 \
      --output-dir outputs/semantic_stress

  python semantic_stress_test.py \
      --real-only --real-logits-npy outputs/mdlm_logits.npy --real-top-k 512 \
      --n-vectors 32 --n-samples 32768 \
      --output-dir outputs/semantic_stress_real

  python semantic_stress_test.py \
      --real-only --checkpoint checkpoints/mdlm.ckpt --text-file validation.txt \
      --real-top-k 512 --n-vectors 32 --n-samples 32768 \
      --output-dir outputs/semantic_stress_checkpoint

Outputs are ``raw_metrics.csv``, ``summary.csv``, ``report.json``, and two PNG
figures. Use ``--no-plots`` on a headless/minimal environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import torch


METRIC_NAMES = (
    "total_variation",
    "max_abs_error",
    "kl_target_empirical",
    "jensen_shannon",
    "top1_probability_error",
)


@dataclass
class LogitCase:
    """A batch of logit vectors belonging to one experimental stratum."""

    case_id: str
    regime: str
    variant: str
    logits: torch.Tensor
    metadata: dict[str, object] = field(default_factory=dict)
    coverage_by_temperature: dict[float, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.logits.ndim != 2:
            raise ValueError(
                f"logits must have shape (n_vectors, vocab), got {self.logits.shape}"
            )
        if self.logits.shape[1] < 2:
            raise ValueError("at least two vocabulary entries are required")
        if not torch.isfinite(self.logits).all():
            raise ValueError(f"case {self.case_id} contains non-finite logits")


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF without requiring SciPy at runtime."""

    flat = x.reshape(-1)
    values = np.fromiter(
        (0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in flat),
        dtype=np.float64,
        count=flat.size,
    )
    return values.reshape(x.shape)


def gaussian_reference_win_probability(
    sigma: float,
    dimensionless_gap: float,
    vocab_size: int,
    quadrature_order: int = 64,
) -> float:
    """Returns a Gaussian noisy-argmax winner probability.

    The reference logits have one coordinate at ``dimensionless_gap`` and the
    other ``vocab_size - 1`` coordinates at zero. Conditioned on the favored
    coordinate's standard-normal noise Z, it wins with probability
    ``Phi(Z + gap / sigma) ** (vocab_size - 1)``. Gauss-Hermite quadrature
    evaluates the remaining one-dimensional expectation deterministically.

    Args:
        sigma: Per-coordinate Gaussian noise standard deviation.
        dimensionless_gap: Reference logit gap after temperature scaling.
        vocab_size: Number of competing coordinates.
        quadrature_order: Gauss-Hermite quadrature order.

    Returns:
        Probability that the favored coordinate wins.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if dimensionless_gap <= 0:
        raise ValueError("dimensionless_gap must be positive")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least two")

    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    standard_normal_nodes = math.sqrt(2.0) * nodes
    cdf = _normal_cdf(standard_normal_nodes + dimensionless_gap / sigma)
    cdf = np.clip(cdf, np.finfo(np.float64).tiny, 1.0)
    values = np.exp((vocab_size - 1) * np.log(cdf))
    return float(np.dot(weights, values) / math.sqrt(math.pi))


def calibrate_gaussian_scale(
    vocab_size: int,
    dimensionless_gap: float,
    tolerance: float = 1e-10,
) -> tuple[float, float, float]:
    """Calibrates Gaussian noise to the canonical one-hot reference.

    Args:
        vocab_size: Number of competing coordinates.
        dimensionless_gap: Reference gap after division by temperature.
        tolerance: Absolute bisection tolerance on the winner probability.

    Returns:
        A tuple ``(sigma, target_probability, calibrated_probability)``.
    """

    target = 1.0 / (1.0 + (vocab_size - 1) * math.exp(-dimensionless_gap))
    lower = 1e-5
    upper = 1.0
    while (
        gaussian_reference_win_probability(upper, dimensionless_gap, vocab_size)
        > target
    ):
        upper *= 2.0
        if upper > 1e6:
            raise RuntimeError("failed to bracket Gaussian calibration scale")

    calibrated = math.nan
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        calibrated = gaussian_reference_win_probability(
            midpoint, dimensionless_gap, vocab_size
        )
        if abs(calibrated - target) <= tolerance:
            return midpoint, target, calibrated
        if calibrated > target:
            lower = midpoint
        else:
            upper = midpoint
    midpoint = (lower + upper) / 2.0
    calibrated = gaussian_reference_win_probability(
        midpoint, dimensionless_gap, vocab_size
    )
    return midpoint, target, calibrated


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little") % (2**63 - 1)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def sample_empirical_distribution(
    scaled_logits: torch.Tensor,
    method: str,
    n_samples: int,
    draw_chunk_size: int,
    generator: torch.Generator,
    gaussian_scale: float | None = None,
) -> torch.Tensor:
    """Samples empirical categorical laws for a batch of logit vectors.

    Args:
        scaled_logits: Tensor with shape ``(batch, vocab)`` containing logits
            already divided by temperature.
        method: One of ``categorical``, ``gumbel``, or ``gaussian``.
        n_samples: Number of samples per vector.
        draw_chunk_size: Maximum draws materialized at once.
        generator: Device-matched random generator.
        gaussian_scale: Calibrated Gaussian noise scale when needed.

    Returns:
        Empirical probabilities with the same shape as ``scaled_logits``.
    """

    if scaled_logits.ndim != 2:
        raise ValueError("scaled_logits must have shape (batch, vocab)")
    if n_samples <= 0 or draw_chunk_size <= 0:
        raise ValueError("sample and chunk counts must be positive")
    if method not in {"categorical", "gumbel", "gaussian"}:
        raise ValueError(f"unknown sampling method: {method}")
    if method == "gaussian" and gaussian_scale is None:
        raise ValueError("gaussian_scale is required for Gaussian sampling")

    batch_size, vocab_size = scaled_logits.shape
    counts = torch.zeros(
        (batch_size, vocab_size), dtype=torch.int64, device=scaled_logits.device
    )
    offsets = (
        torch.arange(batch_size, device=scaled_logits.device, dtype=torch.int64)
        * vocab_size
    )

    if method == "categorical":
        probabilities = scaled_logits.softmax(dim=-1)
        for start in range(0, n_samples, draw_chunk_size):
            draws = min(draw_chunk_size, n_samples - start)
            winners = torch.multinomial(
                probabilities,
                draws,
                replacement=True,
                generator=generator,
            ).transpose(0, 1)
            flat = winners + offsets.unsqueeze(0)
            counts += torch.bincount(
                flat.reshape(-1), minlength=batch_size * vocab_size
            ).reshape(batch_size, vocab_size)
        return counts.to(torch.float64) / n_samples

    for start in range(0, n_samples, draw_chunk_size):
        draws = min(draw_chunk_size, n_samples - start)
        shape = (draws, batch_size, vocab_size)
        if method == "gumbel":
            uniform = torch.rand(
                shape,
                dtype=scaled_logits.dtype,
                device=scaled_logits.device,
                generator=generator,
            )
            tiny = torch.finfo(uniform.dtype).tiny
            noise = -torch.log(-torch.log(uniform.clamp(min=tiny, max=1 - 1e-7)))
        else:
            noise = torch.randn(
                shape,
                dtype=scaled_logits.dtype,
                device=scaled_logits.device,
                generator=generator,
            ) * float(gaussian_scale)
        winners = (scaled_logits.unsqueeze(0) + noise).argmax(dim=-1)
        flat = winners + offsets.unsqueeze(0)
        counts += torch.bincount(
            flat.reshape(-1), minlength=batch_size * vocab_size
        ).reshape(batch_size, vocab_size)
    return counts.to(torch.float64) / n_samples


def distribution_metrics(
    target: torch.Tensor,
    empirical: torch.Tensor,
    n_samples: int,
) -> dict[str, torch.Tensor]:
    """Computes per-vector discrepancies from the target categorical law."""

    target = target.to(torch.float64)
    empirical = empirical.to(torch.float64)
    difference = empirical - target
    total_variation = 0.5 * difference.abs().sum(dim=-1)
    max_abs_error = difference.abs().max(dim=-1).values

    # Jeffreys smoothing keeps finite-sample KL finite without altering TV.
    pseudocount = 0.5
    smoothed = (empirical * n_samples + pseudocount) / (
        n_samples + pseudocount * empirical.shape[-1]
    )
    midpoint = 0.5 * (target + smoothed)
    kl_target_empirical = (
        target * (target.clamp_min(1e-300).log() - smoothed.log())
    ).sum(dim=-1)
    jensen_shannon = 0.5 * (
        target * (target.clamp_min(1e-300).log() - midpoint.log())
    ).sum(dim=-1) + 0.5 * (smoothed * (smoothed.log() - midpoint.log())).sum(dim=-1)

    top1 = target.argmax(dim=-1, keepdim=True)
    top1_probability_error = (
        (empirical.gather(-1, top1) - target.gather(-1, top1)).squeeze(-1).abs()
    )
    entropy = -(target * target.clamp_min(1e-300).log()).sum(dim=-1)
    return {
        "total_variation": total_variation,
        "max_abs_error": max_abs_error,
        "kl_target_empirical": kl_target_empirical,
        "jensen_shannon": jensen_shannon,
        "top1_probability_error": top1_probability_error,
        "target_entropy": entropy,
        "target_effective_support": entropy.exp(),
    }


def _random_case_generator(seed: int, *parts: object) -> torch.Generator:
    return torch.Generator().manual_seed(_stable_seed(seed, *parts))


def build_synthetic_cases(args: argparse.Namespace) -> list[LogitCase]:
    """Builds all requested synthetic logit regimes."""

    cases: list[LogitCase] = []
    for vocab_size in args.vocab_sizes:
        if vocab_size < 2:
            raise ValueError("all vocabulary sizes must be at least two")

        if "one_hot" in args.regimes:
            generator = _random_case_generator(args.seed, "one_hot", vocab_size)
            winners = torch.randint(vocab_size, (args.n_vectors,), generator=generator)
            logits = torch.zeros((args.n_vectors, vocab_size))
            logits.scatter_(1, winners.unsqueeze(-1), args.reference_gap)
            cases.append(
                LogitCase(
                    case_id=f"one_hot_v{vocab_size}",
                    regime="one_hot",
                    variant=f"gap={args.reference_gap:g}",
                    logits=logits,
                    metadata={"reference_gap": args.reference_gap},
                )
            )

        if "random" in args.regimes:
            for scale in args.logit_scales:
                generator = _random_case_generator(
                    args.seed, "random", vocab_size, scale
                )
                logits = (
                    torch.randn((args.n_vectors, vocab_size), generator=generator)
                    * scale
                )
                logits -= logits.mean(dim=-1, keepdim=True)
                cases.append(
                    LogitCase(
                        case_id=f"random_v{vocab_size}_s{scale:g}",
                        regime="random",
                        variant=f"scale={scale:g}",
                        logits=logits,
                        metadata={"logit_scale": scale},
                    )
                )

        if "interpolated" in args.regimes:
            generator = _random_case_generator(args.seed, "interpolated", vocab_size)
            first = torch.randint(vocab_size, (args.n_vectors,), generator=generator)
            offset = torch.randint(
                1, vocab_size, (args.n_vectors,), generator=generator
            )
            second = (first + offset) % vocab_size
            for weight in args.interpolation_weights:
                if not 0.0 <= weight <= 1.0:
                    raise ValueError("interpolation weights must lie in [0, 1]")
                logits = torch.zeros((args.n_vectors, vocab_size))
                logits.scatter_(
                    1,
                    first.unsqueeze(-1),
                    torch.full((args.n_vectors, 1), args.reference_gap * (1 - weight)),
                )
                logits.scatter_(
                    1,
                    second.unsqueeze(-1),
                    torch.full((args.n_vectors, 1), args.reference_gap * weight),
                )
                cases.append(
                    LogitCase(
                        case_id=f"interpolated_v{vocab_size}_w{weight:g}",
                        regime="interpolated",
                        variant=f"weight={weight:g}",
                        logits=logits,
                        metadata={"interpolation_weight": weight},
                    )
                )

        if "guided" in args.regimes:
            generator = _random_case_generator(args.seed, "guided", vocab_size)
            unconditional = torch.randn(
                (args.n_vectors, vocab_size), generator=generator
            )
            residual = torch.randn((args.n_vectors, vocab_size), generator=generator)
            residual -= residual.mean(dim=-1, keepdim=True)
            residual /= residual.std(dim=-1, keepdim=True).clamp_min(1e-6)
            conditional = unconditional + args.guidance_residual_scale * residual
            for guidance_scale in args.guidance_scales:
                logits = unconditional + guidance_scale * (conditional - unconditional)
                logits -= logits.mean(dim=-1, keepdim=True)
                cases.append(
                    LogitCase(
                        case_id=f"guided_v{vocab_size}_g{guidance_scale:g}",
                        regime="guided",
                        variant=f"guidance={guidance_scale:g}",
                        logits=logits,
                        metadata={"guidance_scale": guidance_scale},
                    )
                )
    return cases


def _prepare_real_case(
    logits: torch.Tensor,
    case_id: str,
    variant: str,
    args: argparse.Namespace,
    metadata: dict[str, object] | None = None,
) -> LogitCase:
    """Cleans, subsamples, and optionally top-k truncates real logits."""

    logits = logits.reshape(-1, logits.shape[-1]).to(torch.float32)
    if logits.shape[0] < args.n_vectors:
        raise ValueError(
            f"{case_id} has {logits.shape[0]} vectors; need {args.n_vectors}"
        )

    # SUBS models commonly emit a permanently impossible mask column near
    # -1e6. It is not part of the clean vocabulary and must not affect the
    # Gaussian calibration dimension.
    relative = logits - logits.max(dim=-1, keepdim=True).values
    usable_columns = torch.isfinite(relative).all(dim=0) & (
        relative.max(dim=0).values > -1e4
    )
    logits = logits[:, usable_columns]
    if logits.shape[1] < 2:
        raise ValueError("fewer than two usable real-logit columns remain")

    generator = _random_case_generator(args.seed, "real", case_id)
    selected_rows = torch.randperm(logits.shape[0], generator=generator)[
        : args.n_vectors
    ]
    logits = logits[selected_rows]
    logits -= logits.max(dim=-1, keepdim=True).values

    coverage_by_temperature: dict[float, torch.Tensor] = {}
    original_vocab_size = logits.shape[1]
    if 0 < args.real_top_k < logits.shape[1]:
        top_logits, top_indices = logits.topk(args.real_top_k, dim=-1)
        for temperature in args.temperatures:
            full_probabilities = (logits / temperature).softmax(dim=-1)
            coverage_by_temperature[float(temperature)] = full_probabilities.gather(
                -1, top_indices
            ).sum(dim=-1)
        logits = top_logits
    else:
        for temperature in args.temperatures:
            coverage_by_temperature[float(temperature)] = torch.ones(logits.shape[0])

    case_metadata = dict(metadata or {})
    case_metadata.update(
        {
            "original_vocab_size": original_vocab_size,
            "evaluated_vocab_size": logits.shape[1],
        }
    )
    return LogitCase(
        case_id=case_id,
        regime="denoiser",
        variant=variant,
        logits=logits,
        metadata=case_metadata,
        coverage_by_temperature=coverage_by_temperature,
    )


def load_real_logits(args: argparse.Namespace) -> list[LogitCase]:
    """Loads a NumPy logit array with shape ``(..., vocab)``."""

    path = Path(args.real_logits_npy)
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "logits" in loaded.files:
            array = np.asarray(loaded["logits"])
        elif len(loaded.files) == 1:
            array = np.asarray(loaded[loaded.files[0]])
        else:
            loaded.close()
            raise ValueError("NPZ must contain a 'logits' array or exactly one array")
        loaded.close()
    else:
        array = loaded
    if array.ndim < 2:
        raise ValueError("real logits must have shape (..., vocab)")
    return [
        _prepare_real_case(
            torch.from_numpy(np.asarray(array)),
            case_id="denoiser_numpy",
            variant=path.name,
            args=args,
            metadata={"source": str(path)},
        )
    ]


def _iter_texts(args: argparse.Namespace) -> Iterator[str]:
    """Yields local text or streams a Hugging Face dataset."""

    if args.text_file:
        with Path(args.text_file).open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield line
        return

    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        streaming=True,
    )
    for example in dataset:
        text = example.get(args.dataset_text_field)
        if text:
            yield text


def _token_sequences(
    texts: Iterable[str], tokenizer: object, seq_len: int
) -> Iterator[list[int]]:
    """Packs a text iterator into fixed-length token sequences."""

    buffer: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for text in texts:
        buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if eos_token_id is not None:
            buffer.append(eos_token_id)
        while len(buffer) >= seq_len:
            yield buffer[:seq_len]
            del buffer[:seq_len]


def collect_checkpoint_logits(args: argparse.Namespace) -> list[LogitCase]:
    """Collects masked-position clean-token logits from an MDLM checkpoint."""

    # Reuse the repository's checkpoint construction so E1 and E3 load models
    # identically. These heavyweight imports remain local because the fully
    # synthetic experiment does not need Hydra or Transformers.
    from curvature_decode import build_model

    model_args = argparse.Namespace(
        model="ckpt", checkpoint=args.checkpoint, device=args.device
    )
    model = build_model(model_args)
    device = torch.device(args.device)
    generator = _make_generator(
        device, _stable_seed(args.seed, "checkpoint_corruption")
    )
    sequences = _token_sequences(_iter_texts(args), model.tokenizer, args.real_seq_len)
    collected: dict[float, list[torch.Tensor]] = {
        float(mask_fraction): [] for mask_fraction in args.mask_fractions
    }
    totals = {float(mask_fraction): 0 for mask_fraction in args.mask_fractions}

    while min(totals.values()) < args.n_vectors:
        rows = []
        for _ in range(args.checkpoint_batch_size):
            try:
                rows.append(next(sequences))
            except StopIteration as error:
                raise ValueError(
                    "text source ended before enough denoiser logits were collected"
                ) from error
        clean = torch.tensor(rows, dtype=torch.long, device=device)
        for mask_fraction in args.mask_fractions:
            key = float(mask_fraction)
            if totals[key] >= args.n_vectors:
                continue
            masked = torch.rand(
                clean.shape, device=device, generator=generator
            ) < float(mask_fraction)
            masked[:, 0] = False
            noisy = torch.where(masked, model.mask_index, clean)
            alpha = torch.full(
                (clean.shape[0], 1),
                1.0 - float(mask_fraction),
                device=device,
            )
            sigma = model._sigma_from_alphat(alpha)
            with torch.no_grad():
                log_probabilities = model.forward(noisy, sigma=sigma)
            clean_columns = torch.arange(model.vocab_size, device=device)
            clean_columns = clean_columns[clean_columns != model.mask_index]
            vectors = log_probabilities[..., clean_columns][masked].detach().cpu()
            needed = args.n_vectors - totals[key]
            vectors = vectors[:needed]
            collected[key].append(vectors)
            totals[key] += vectors.shape[0]

    cases = []
    for mask_fraction, chunks in collected.items():
        cases.append(
            _prepare_real_case(
                torch.cat(chunks, dim=0),
                case_id=f"denoiser_mask{mask_fraction:g}",
                variant=f"mask_fraction={mask_fraction:g}",
                args=args,
                metadata={
                    "source": args.checkpoint,
                    "mask_fraction": mask_fraction,
                },
            )
        )
    return cases


def _case_rows(
    case: LogitCase,
    temperature: float,
    methods: Sequence[str],
    gaussian_scale: float,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Evaluates all methods for one case and temperature."""

    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    for start in range(0, case.logits.shape[0], args.vector_batch_size):
        logits = case.logits[start : start + args.vector_batch_size].to(device)
        scaled_logits = logits / temperature
        target = scaled_logits.softmax(dim=-1).to(torch.float64)
        for method in methods:
            seed = _stable_seed(args.seed, case.case_id, temperature, method, start)
            generator = _make_generator(device, seed)
            empirical = sample_empirical_distribution(
                scaled_logits=scaled_logits,
                method=method,
                n_samples=args.n_samples,
                draw_chunk_size=args.draw_chunk_size,
                generator=generator,
                gaussian_scale=gaussian_scale,
            )
            metrics = distribution_metrics(target, empirical, args.n_samples)
            for local_index in range(logits.shape[0]):
                vector_index = start + local_index
                row: dict[str, object] = {
                    "case_id": case.case_id,
                    "vector_id": vector_index,
                    "regime": case.regime,
                    "variant": case.variant,
                    "vocab_size": logits.shape[-1],
                    "temperature": temperature,
                    "method": method,
                    "n_samples": args.n_samples,
                    "gaussian_scale": gaussian_scale,
                    "topk_target_mass": float(
                        case.coverage_by_temperature.get(
                            float(temperature), torch.ones(case.logits.shape[0])
                        )[vector_index]
                    ),
                }
                row.update(
                    {
                        name: float(values[local_index].cpu())
                        for name, values in metrics.items()
                    }
                )
                rows.append(row)
    return rows


def _add_monte_carlo_floor(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["case_id", "vector_id", "temperature"]
    floor_columns = keys + list(METRIC_NAMES)
    floor = raw.loc[raw["method"] == "categorical", floor_columns].copy()
    floor = floor.rename(columns={name: f"{name}_mc_floor" for name in METRIC_NAMES})
    merged = raw.merge(floor, on=keys, how="left", validate="many_to_one")
    for name in METRIC_NAMES:
        merged[f"{name}_excess"] = merged[name] - merged[f"{name}_mc_floor"]
    return merged


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregates per-vector results and adds 95% normal-approximation CIs."""

    group_columns = [
        "regime",
        "variant",
        "vocab_size",
        "temperature",
        "method",
    ]
    metrics = [*METRIC_NAMES, *(f"{name}_excess" for name in METRIC_NAMES)]
    records: list[dict[str, object]] = []
    for keys, group in raw.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["n_vectors"] = len(group)
        row["topk_target_mass_mean"] = group["topk_target_mass"].mean()
        for metric in metrics:
            values = group[metric]
            standard_error = values.std(ddof=1) / math.sqrt(len(values))
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_p90"] = values.quantile(0.9)
            row[f"{metric}_ci95"] = (
                0.0 if math.isnan(standard_error) else 1.96 * standard_error
            )
        records.append(row)
    return pd.DataFrame.from_records(records)


def _save_plots(raw: pd.DataFrame, output_dir: Path) -> None:
    # Plotting is optional, so keep these heavyweight imports off the synthetic
    # smoke-test path and allow the core experiment to run in minimal jobs.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")

    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=raw,
        x="regime",
        y="max_abs_error",
        hue="method",
        errorbar=("ci", 95),
        ax=axis,
    )
    axis.set_yscale("log")
    axis.set_ylabel("Maximum probability error")
    axis.set_xlabel("Logit regime")
    axis.set_title("Exact categorical semantics beyond calibration states")
    figure.tight_layout()
    figure.savefig(output_dir / "max_error_by_regime.png", dpi=180)
    plt.close(figure)

    plot_data = raw.loc[raw["method"] != "categorical"].copy()
    grid = sns.relplot(
        data=plot_data,
        x="vocab_size",
        y="max_abs_error_excess",
        hue="method",
        col="regime",
        col_wrap=2,
        kind="line",
        marker="o",
        errorbar=("ci", 95),
        facet_kws={"sharey": False},
        height=4.0,
        aspect=1.25,
    )
    grid.set(xscale="log")
    grid.set_axis_labels("Vocabulary size", "Excess max error over MC floor")
    grid.figure.suptitle(
        "Departure from softmax semantics after one-hot calibration", y=1.03
    )
    grid.figure.savefig(
        output_dir / "excess_error_vs_vocab.png", dpi=180, bbox_inches="tight"
    )
    plt.close(grid.figure)


def _headline(raw: pd.DataFrame) -> dict[str, object]:
    one_hot = raw.loc[raw["regime"] == "one_hot"]
    soft = raw.loc[raw["regime"] != "one_hot"]

    def mean_for(frame: pd.DataFrame, method: str, metric: str) -> float | None:
        values = frame.loc[frame["method"] == method, metric]
        return None if values.empty else float(values.mean())

    return {
        "one_hot_mean_max_error": {
            method: mean_for(one_hot, method, "max_abs_error")
            for method in raw["method"].unique()
        },
        "non_one_hot_mean_max_error": {
            method: mean_for(soft, method, "max_abs_error")
            for method in raw["method"].unique()
        },
        "non_one_hot_mean_excess_max_error": {
            method: mean_for(soft, method, "max_abs_error_excess")
            for method in raw["method"].unique()
        },
    }


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """Runs E1 and writes all outputs."""

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if args.n_vectors <= 0 or args.n_samples <= 0:
        raise ValueError("n_vectors and n_samples must be positive")
    if args.reference_gap <= 0:
        raise ValueError("reference_gap must be positive")
    if any(temperature <= 0 for temperature in args.temperatures):
        raise ValueError("temperatures must be positive")
    if "categorical" not in args.methods:
        raise ValueError("methods must include categorical as the MC floor")

    cases = [] if args.real_only else build_synthetic_cases(args)
    if args.real_logits_npy:
        cases.extend(load_real_logits(args))
    if args.checkpoint:
        cases.extend(collect_checkpoint_logits(args))
    if not cases:
        raise ValueError("no synthetic or real-logit cases were requested")

    calibrations: dict[tuple[int, float], tuple[float, float, float]] = {}
    calibration_rows = []
    for vocab_size in sorted({case.logits.shape[1] for case in cases}):
        for temperature in args.temperatures:
            dimensionless_gap = args.reference_gap / temperature
            calibration = calibrate_gaussian_scale(vocab_size, dimensionless_gap)
            calibrations[(vocab_size, float(temperature))] = calibration
            sigma, target_probability, calibrated_probability = calibration
            calibration_rows.append(
                {
                    "vocab_size": vocab_size,
                    "temperature": temperature,
                    "reference_gap": args.reference_gap,
                    "dimensionless_gap": dimensionless_gap,
                    "gaussian_scale": sigma,
                    "target_winner_probability": target_probability,
                    "calibrated_winner_probability": calibrated_probability,
                    "absolute_residual": abs(
                        target_probability - calibrated_probability
                    ),
                }
            )

    rows: list[dict[str, object]] = []
    for case in cases:
        for temperature in args.temperatures:
            gaussian_scale = calibrations[(case.logits.shape[1], float(temperature))][0]
            rows.extend(
                _case_rows(
                    case,
                    float(temperature),
                    args.methods,
                    gaussian_scale,
                    args,
                )
            )
            print(
                f"[e1] {case.case_id} temperature={temperature:g} complete",
                flush=True,
            )

    raw = _add_monte_carlo_floor(pd.DataFrame.from_records(rows))
    summary = summarize(raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "raw_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame.from_records(calibration_rows).to_csv(
        output_dir / "gaussian_calibration.csv", index=False
    )
    if not args.no_plots:
        _save_plots(raw, output_dir)

    headline = _headline(raw)
    report = {
        "experiment": "E1 exact softmax semantics stress test",
        "config": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, list, type(None)))
        },
        "cases": [
            {
                "case_id": case.case_id,
                "regime": case.regime,
                "variant": case.variant,
                "n_vectors": case.logits.shape[0],
                "vocab_size": case.logits.shape[1],
                "metadata": case.metadata,
            }
            for case in cases
        ],
        "calibration": calibration_rows,
        "headline": headline,
        "interpretation_rule": (
            "Gumbel supports the arbitrary-logit semantics claim when its "
            "non-one-hot excess error remains near the categorical Monte Carlo "
            "floor while calibrated Gaussian excess error is positive. The "
            "one-hot stratum checks that the Gaussian control was calibrated "
            "fairly."
        ),
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(headline, indent=2))
    print(f"[e1] wrote results to {output_dir}")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=["one_hot", "random", "interpolated", "guided"],
        default=["one_hot", "random", "interpolated", "guided"],
    )
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[2, 16, 128])
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    parser.add_argument("--n-vectors", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=8192)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["categorical", "gumbel", "gaussian"],
        default=["categorical", "gumbel", "gaussian"],
    )
    parser.add_argument("--reference-gap", type=float, default=4.0)
    parser.add_argument("--logit-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--interpolation-weights", type=float, nargs="+", default=[0.5])
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[2.0])
    parser.add_argument("--guidance-residual-scale", type=float, default=0.5)
    parser.add_argument("--draw-chunk-size", type=int, default=256)
    parser.add_argument("--vector-batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/semantic_stress")
    parser.add_argument("--no-plots", action="store_true")

    real_group = parser.add_argument_group("real denoiser logits")
    real_group.add_argument(
        "--real-only",
        action="store_true",
        help="Skip synthetic cases and evaluate only the supplied real logits.",
    )
    real_group.add_argument("--real-logits-npy", default=None)
    real_group.add_argument(
        "--real-top-k",
        type=int,
        default=512,
        help="Evaluate the top-k subdistribution; <=0 keeps the full vocabulary.",
    )
    real_group.add_argument("--checkpoint", default=None)
    real_group.add_argument("--text-file", default=None)
    real_group.add_argument("--dataset-name", default="openwebtext")
    real_group.add_argument("--dataset-split", default="train")
    real_group.add_argument("--dataset-text-field", default="text")
    real_group.add_argument(
        "--mask-fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75]
    )
    real_group.add_argument("--real-seq-len", type=int, default=128)
    real_group.add_argument("--checkpoint-batch-size", type=int, default=8)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny CPU configuration to verify plumbing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.real_logits_npy and args.checkpoint:
        parser.error("choose at most one of --real-logits-npy and --checkpoint")
    if any(not 0.0 < fraction < 1.0 for fraction in args.mask_fractions):
        parser.error("mask fractions must lie strictly between zero and one")
    if args.smoke:
        args.real_only = False
        args.regimes = ["one_hot", "random", "interpolated"]
        args.vocab_sizes = [4]
        args.temperatures = [1.0]
        args.n_vectors = 4
        args.n_samples = 1024
        args.logit_scales = [1.0]
        args.interpolation_weights = [0.5]
        args.draw_chunk_size = 128
        args.vector_batch_size = 2
        args.device = "cpu"
    return run_experiment(args)


if __name__ == "__main__":
    main()
