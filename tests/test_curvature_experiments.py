"""Functional regression tests for curvature validation and decoding."""

from argparse import Namespace
import math
from pathlib import Path
import sys

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import curvature_decode
import curvature_validate


def test_evaluate_accepts_bfloat16_and_respects_target_validity():
    scores = {
        "ordered": torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, -100, 100]],
            dtype=torch.bfloat16,
        )
    }
    targets = {
        "stability": torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 1_000, -1_000]],
            dtype=torch.bfloat16,
        )
    }
    validity = {
        "stability": torch.tensor(
            [[True, True, True, True, True, True, True, True, False, False]]
        )
    }

    report = curvature_validate.evaluate(scores, targets, validity)

    assert report["stability"]["ordered"]["spearman"] == 1.0


def test_candidate_mask_is_capped_and_reproducible():
    masked = torch.tensor([[True, True, True, True], [False, True, True, False]])
    first_generator = torch.Generator().manual_seed(11)
    second_generator = torch.Generator().manual_seed(11)

    first = curvature_validate.candidate_mask(masked, 2, first_generator)
    second = curvature_validate.candidate_mask(masked, 2, second_generator)

    assert torch.equal(first, second)
    assert bool((first <= masked).all())
    assert first.sum(dim=-1).tolist() == [2, 2]


def test_random_plumbing_batch_never_emits_nonterminal_mask_token():
    model = _StabilityModel()

    tokens, source = curvature_validate.real_batch(
        model,
        n=8,
        L=16,
        device="cpu",
        seed=4,
        data_source="random",
    )

    assert source == "random"
    assert not bool((tokens == model.mask_index).any())


class _StabilityModel:
    vocab_size = 4
    mask_index = 1

    def _sigma_from_alphat(self, alpha):
        return alpha

    def forward(self, tokens, sigma):
        del sigma
        batch_size, length = tokens.shape
        logits = torch.zeros((batch_size, length, self.vocab_size))
        logits[..., self.mask_index] = -1e6
        # Once clean token 2 is visible elsewhere, the masked position changes
        # its preferred token from 0 to 2.
        resolved = (tokens == 2).any(dim=-1)
        for batch_index in range(batch_size):
            if resolved[batch_index]:
                logits[batch_index, :, 2] = 3
            else:
                logits[batch_index, :, 0] = 3
        return logits.log_softmax(dim=-1).to(torch.bfloat16)


def test_stability_reports_actual_agreement_and_probability():
    model = _StabilityModel()
    clean = torch.tensor([[0, 2]])
    masked = torch.tensor([[True, True]])
    noisy = torch.full_like(clean, model.mask_index)
    initial_logp = model.forward(noisy, sigma=torch.ones((1, 1)))
    _, initial_prediction = curvature_validate.initial_targets(
        model, clean, initial_logp
    )
    selected = torch.tensor([[True, False]])

    stability = curvature_validate._leave_one_out_stability(
        model,
        noisy,
        clean,
        masked,
        initial_prediction,
        selected,
    )

    assert stability["stability"][0, 0].item() == 0
    assert 0 < stability["stability_probability"][0, 0].item() < 0.1
    # An unselected position remains storage-only and must be excluded by its
    # validity mask in the caller.
    assert stability["stability"][0, 1].item() == 0


class _ContinuousInputModel:
    vocab_size = 3
    mask_index = 1

    def __init__(self):
        self.forward_calls = 0

    @property
    def backbone(self):
        raise AssertionError("curvature must use processed model.forward")

    def forward(self, tokens, sigma, nn_input_idxs=None):
        del sigma
        self.forward_calls += 1
        assert nn_input_idxs is not None
        logits = nn_input_idxs.clone()
        logits[..., self.mask_index] = -1e6
        return logits.log_softmax(dim=-1).to(torch.bfloat16)


def test_sensitivity_uses_processed_forward_and_returns_float32():
    model = _ContinuousInputModel()
    tokens = torch.tensor([[1, 0]])
    masked = tokens == model.mask_index
    generator = torch.Generator().manual_seed(3)

    score = curvature_decode.sensitivity_score(
        model,
        tokens,
        sigma=torch.ones((1, 1)),
        masked=masked,
        n_probes=2,
        eps=0.1,
        in_scale=2.0,
        generator=generator,
    )

    assert model.forward_calls == 3
    assert score.dtype == torch.float32
    assert torch.isfinite(score).all()


def _gumbel_log_marginal(observed_logits, prior, signal_scale, noise_scale):
    vocabulary_size = observed_logits.shape[-1]
    means = signal_scale * torch.eye(
        vocabulary_size,
        dtype=observed_logits.dtype,
        device=observed_logits.device,
    )
    residual = observed_logits - means
    component_log_density = (
        -torch.log(torch.as_tensor(noise_scale, dtype=observed_logits.dtype))
        - residual / noise_scale
        - torch.exp(-residual / noise_scale)
    ).sum(dim=-1)
    return torch.logsumexp(prior.log() + component_log_density, dim=0)


def test_exact_gumbel_tweedie_terms_match_autograd_mixture_hessian():
    prior = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    observed = torch.tensor(
        [0.2, -0.4, 0.7, 0.1], dtype=torch.float64, requires_grad=True
    )
    signal_scale = 1.3
    noise_scale = 0.8

    log_marginal = _gumbel_log_marginal(observed, prior, signal_scale, noise_scale)
    marginal_score = torch.autograd.grad(log_marginal, observed, create_graph=True)[0]
    marginal_hessian = torch.autograd.functional.hessian(
        lambda value: _gumbel_log_marginal(value, prior, signal_scale, noise_scale),
        observed,
    )

    means = signal_scale * torch.eye(4, dtype=torch.float64)
    residual = observed.detach() - means
    component_log_density = (
        -math.log(noise_scale)
        - residual / noise_scale
        - torch.exp(-residual / noise_scale)
    ).sum(dim=-1)
    posterior_logp = torch.log_softmax(prior.log() + component_log_density, dim=-1)
    terms = curvature_decode.gumbel_tweedie_terms(
        posterior_logp,
        signal_scale=signal_scale,
        noise_scale=noise_scale,
        observed_logits=observed.detach(),
    )

    assert torch.allclose(
        terms.marginal_score, marginal_score.detach().float(), atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        terms.marginal_hessian_diag,
        marginal_hessian.diag().float(),
        atol=2e-5,
        rtol=2e-5,
    )
    assert terms.marginal_hessian_trace.item() == pytest.approx(
        marginal_hessian.trace().item(), abs=2e-5
    )
    probe = torch.tensor([0.3, -0.5, 1.1, 0.2], dtype=torch.float64)
    hessian_product = curvature_decode.gumbel_tweedie_hessian_vector_product(
        posterior_logp,
        probe,
        signal_scale=signal_scale,
        noise_scale=noise_scale,
        observed_logits=observed.detach(),
    )
    assert torch.allclose(
        hessian_product,
        (marginal_hessian @ probe).float(),
        atol=2e-5,
        rtol=2e-5,
    )


def test_neutral_tweedie_curvature_is_scaled_clean_gini_with_bfloat16_input():
    probabilities = torch.tensor([0.2, 0.0, 0.3, 0.5], dtype=torch.float32)
    logp = probabilities.clamp_min(1e-30).log().to(torch.bfloat16)
    signal_scale = 1.2
    noise_scale = 0.9

    terms = curvature_decode.gumbel_tweedie_terms(
        logp,
        mask_index=1,
        signal_scale=signal_scale,
        noise_scale=noise_scale,
    )
    clean_gini = 1.0 - probabilities.square().sum()
    multiplier = math.expm1(signal_scale / noise_scale) ** 2 / noise_scale**2

    assert terms.score_covariance_trace.dtype == torch.float32
    assert terms.score_covariance_trace.item() == pytest.approx(
        multiplier * clean_gini.item(), rel=3e-3
    )
    assert terms.score_covariance_diag[1].item() == 0


def test_exact_curvature_score_is_commit_first_negative_covariance():
    probabilities = torch.tensor(
        [
            [[0.49, 0.0, 0.49, 0.02], [0.9, 0.0, 0.05, 0.05]],
        ]
    )
    logp = probabilities.clamp_min(1e-30).log()

    score = curvature_decode.curvature_score(logp, mask_index=1)

    assert score.dtype == torch.float32
    assert score[0, 1] > score[0, 0]


def test_sample_clean_tokens_handles_nonterminal_mask_and_is_reproducible():
    logp = torch.log(
        torch.tensor([[0.2, 0.0, 0.3, 0.5]], dtype=torch.float32).clamp_min(1e-30)
    ).repeat(2_000, 1)
    first_generator = torch.Generator().manual_seed(5)
    second_generator = torch.Generator().manual_seed(5)

    first = curvature_decode.sample_clean_tokens(logp, 1, first_generator)
    second = curvature_decode.sample_clean_tokens(logp, 1, second_generator)

    assert torch.equal(first, second)
    assert not bool((first == 1).any())
    assert (first == 3).float().mean().item() == pytest.approx(0.5, abs=0.04)


class _DecodeModel:
    vocab_size = 4
    mask_index = 1

    def _sigma_from_alphat(self, alpha):
        return alpha

    def forward(self, tokens, sigma):
        del sigma
        probabilities = torch.tensor([0.2, 0.0, 0.3, 0.5], dtype=torch.float32)
        return (
            probabilities.clamp_min(1e-30).log().expand(*tokens.shape, self.vocab_size)
        )


def test_decoder_samples_diverse_clean_sequences_instead_of_argmax_duplicates():
    args = Namespace(
        device="cpu",
        batch_size=32,
        n_probes=1,
        eps=0.1,
        in_scale=2.0,
    )
    token_generator = torch.Generator().manual_seed(13)
    score_generator = torch.Generator().manual_seed(14)
    probe_generator = torch.Generator().manual_seed(15)

    tokens, _, _ = curvature_decode.decode(
        _DecodeModel(),
        order="confidence",
        n_steps=2,
        n_samples=32,
        L=4,
        args=args,
        token_gen=token_generator,
        score_gen=score_generator,
        probe_gen=probe_generator,
    )

    assert not bool((tokens == _DecodeModel.mask_index).any())
    assert torch.unique(tokens, dim=0).shape[0] > 1
