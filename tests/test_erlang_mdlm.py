"""Functional tests for the Erlang-k augmented masked diffusion model."""

import sys
from pathlib import Path

import hydra
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import algo
import trainer_base


_BATCH_SIZE = 2
_SEQUENCE_LENGTH = 16
_NUM_PHASE_SAMPLES = 300_000


class _DummyMetrics(torch.nn.Module):
    """Avoids loading external evaluation models in unit tests."""

    def __init__(self, *args, **kwargs):
        del args, kwargs
        super().__init__()


class _DummyTokenizer:
    """Minimal tokenizer contract needed by AbsorbingState."""

    vocab_size = 31
    mask_token = None
    pad_token_id = 0

    def __len__(self):
        return self.vocab_size


@pytest.fixture
def build_model(monkeypatch):
    """Builds tiny diffusion models without network-dependent metrics."""
    monkeypatch.setattr(trainer_base.metrics, "Metrics", _DummyMetrics)

    def _build(erlang_k=1, algorithm="erlang", extra_overrides=None):
        overrides = [
            f"algo={algorithm}",
            "model=tiny",
            "model.hidden_size=32",
            "model.cond_dim=32",
            "model.n_blocks=1",
            "model.n_heads=4",
            "model.length=16",
            "model.dropout=0.0",
            "data=openwebtext-split",
            "loader.batch_size=2",
            "loader.eval_batch_size=2",
            "loader.global_batch_size=2",
            "training.ema=0",
            "trainer.precision=32",
            "sampling.predictor=ancestral_cache",
            "eval.generate_samples=False",
        ]
        if algorithm == "erlang":
            overrides.append(f"algo.erlang_k={erlang_k}")
        else:
            overrides.append("algo.time_conditioning=True")
        overrides.extend(extra_overrides or [])
        with hydra.initialize_config_dir(
            config_dir=str(REPO_ROOT / "configs"), version_base=None
        ):
            config = hydra.compose(config_name="config", overrides=overrides)
        model_class = algo.ErlangMDLM if algorithm == "erlang" else algo.MDLM
        return model_class(config, tokenizer=_DummyTokenizer())

    return _build


@pytest.mark.parametrize("erlang_k", [1, 2, 4, 8])
def test_intensity_matches_mdlm_mask_marginal(build_model, erlang_k):
    model = build_model(erlang_k=erlang_k)
    alpha = torch.tensor([[0.999], [0.8], [0.5], [0.1]])

    intensity = model._erlang_intensity(alpha)

    assert torch.allclose(
        model._erlang_survival(intensity), alpha, atol=2e-6, rtol=2e-6
    )


def test_phase_sampler_uses_sequential_erlang_law(build_model):
    torch.manual_seed(3)
    model = build_model(erlang_k=4)
    alpha = torch.tensor([[0.5]])
    x0 = torch.zeros((256, 1024), dtype=torch.long)

    phase = model._sample_phase(x0, alpha)
    observed = torch.bincount(phase.flatten(), minlength=model.erlang_k + 1).float()
    observed /= phase.numel()
    expected = model._phase_probabilities(model._erlang_intensity(alpha)).flatten()

    assert torch.allclose(observed, expected, atol=0.005, rtol=0.0)
    assert abs(observed[-1].item() - 0.5) < 0.005


def test_masked_reverse_posterior_matches_forward_process(build_model):
    torch.manual_seed(5)
    model = build_model(erlang_k=4)
    alpha_s = torch.tensor([[0.75]])
    alpha_t = torch.tensor([[0.25]])
    intensity_s = model._erlang_intensity(alpha_s)
    intensity_t = model._erlang_intensity(alpha_t)
    expected = model._masked_phase_posterior(intensity_s, intensity_t).flatten()

    phase_s = torch.poisson(intensity_s.flatten().expand(_NUM_PHASE_SAMPLES))
    increment = torch.poisson(
        (intensity_t - intensity_s).flatten().expand(_NUM_PHASE_SAMPLES)
    )
    phase_t = (phase_s + increment).clamp(max=model.erlang_k)
    phase_s = phase_s.clamp(max=model.erlang_k).to(torch.long)
    phase_s = phase_s[phase_t == model.erlang_k]
    observed = torch.bincount(phase_s, minlength=model.erlang_k + 1).float()
    observed /= phase_s.numel()

    assert torch.allclose(expected.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(observed, expected, atol=0.005, rtol=0.0)


def test_k1_logits_match_time_conditioned_mdlm(build_model):
    torch.manual_seed(7)
    erlang = build_model(erlang_k=1)
    mdlm = build_model(algorithm="mdlm")
    torch.nn.init.normal_(erlang.backbone.output_layer.linear.weight, std=0.02)
    torch.nn.init.normal_(erlang.backbone.output_layer.linear.bias, std=0.02)
    mdlm.load_state_dict(erlang.state_dict())
    erlang.eval()
    mdlm.eval()

    x0 = torch.randint(
        0,
        erlang.tokenizer.vocab_size,
        (_BATCH_SIZE, _SEQUENCE_LENGTH),
    )
    phase = torch.randint(0, 2, x0.shape)
    xt = torch.where(phase == 1, torch.full_like(x0, erlang.mask_index), x0)
    alpha = torch.full((_BATCH_SIZE, 1), 0.5)
    dalpha = torch.full((_BATCH_SIZE, 1), -0.999)
    sigma = erlang._sigma_from_alphat(alpha)

    erlang_logits = erlang._phase_logits(xt, phase, sigma)
    mdlm_logits = mdlm.forward(xt, sigma)
    erlang_loss = erlang.nll_per_token(erlang_logits, xt, x0, alpha, dalpha)
    mdlm_loss = mdlm.nll_per_token(mdlm_logits, xt, x0, alpha, dalpha)

    assert torch.allclose(erlang_logits, mdlm_logits, atol=1e-6, rtol=0.0)
    assert torch.allclose(erlang_loss, mdlm_loss, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("erlang_k", [1, 2, 4, 8])
def test_loss_uses_unscaled_mdlm_coefficient(build_model, monkeypatch, erlang_k):
    model = build_model(erlang_k=erlang_k)
    x0 = torch.randint(
        0,
        model.tokenizer.vocab_size,
        (_BATCH_SIZE, _SEQUENCE_LENGTH),
    )
    phase = torch.arange(_BATCH_SIZE * _SEQUENCE_LENGTH).reshape(
        _BATCH_SIZE, _SEQUENCE_LENGTH
    ) % (erlang_k + 1)
    masked = phase == erlang_k
    true_log_probability = -2.5
    log_probabilities = torch.full((*x0.shape, model.vocab_size), model.neg_infinity)
    log_probabilities.scatter_(-1, x0.unsqueeze(-1), true_log_probability)
    visible_indices = (~masked).nonzero(as_tuple=True)
    log_probabilities[visible_indices[0], visible_indices[1], x0[visible_indices]] = 0.0

    monkeypatch.setattr(
        model,
        "_sample_t",
        lambda batch_size, accumulation_step: torch.full((batch_size,), 0.5),
    )
    monkeypatch.setattr(model, "_sample_phase", lambda tokens, alpha: phase)
    monkeypatch.setattr(
        model,
        "_phase_logits",
        lambda tokens, current_phase, sigma: log_probabilities,
    )

    loss = model.nll(x0, None, None)
    dalpha, alpha = model.noise(torch.full((_BATCH_SIZE,), 0.5))
    expected_masked_loss = true_log_probability * (dalpha / (1.0 - alpha))

    assert torch.count_nonzero(loss[~masked]) == 0
    assert torch.allclose(
        loss[masked],
        expected_masked_loss[:, None].expand_as(loss)[masked],
    )


def test_training_updates_phase_embedding_and_sampling_removes_masks(
    build_model, monkeypatch
):
    torch.manual_seed(11)
    model = build_model(erlang_k=4)
    torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.02)
    torch.nn.init.normal_(model.backbone.output_layer.linear.bias, std=0.02)
    monkeypatch.setattr(
        model,
        "_sample_t",
        lambda batch_size, accumulation_step: torch.full((batch_size,), 0.5),
    )
    x0 = torch.randint(0, model.tokenizer.vocab_size, (4, _SEQUENCE_LENGTH))

    loss = model.nll(x0, None, None).mean()
    loss.backward()

    phase_gradient = model.backbone.phase_embed.weight.grad
    assert torch.isfinite(loss)
    assert phase_gradient is not None
    assert torch.isfinite(phase_gradient).all()
    assert phase_gradient.norm() > 0

    model.eval()
    samples = model.generate_samples(num_samples=2, num_steps=4, eps=1e-3)
    assert samples.shape == (2, model.num_tokens)
    assert not bool((samples == model.mask_index).any())


def test_non_dit_backbone_is_rejected_before_loading(build_model):
    with pytest.raises(ValueError, match="requires algo.backbone=dit"):
        build_model(
            erlang_k=2,
            extra_overrides=[
                "algo.backbone=hf_dit",
                "eval.checkpoint_path=unused",
            ],
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            "algo.causal_attention=True",
            "requires causal_attention=False",
        ),
        (
            "sampling.semi_ar=True",
            "does not yet implement semi-autoregressive sampling",
        ),
    ],
)
def test_unsupported_sampling_paths_are_rejected(build_model, override, message):
    with pytest.raises(ValueError, match=message):
        build_model(erlang_k=2, extra_overrides=[override])
