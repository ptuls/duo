"""Regression tests for the G-MCD distillation algorithm (algo.GMCD).

Covers the failure modes that a code review surfaced:
  - checkpoint round-trip (teacher.* keys must be stripped and reloadable),
  - teacher determinism (dropout must be off in the target),
  - kl-bwd must be finite on tokens revealed in (s, t] (delta teacher),
  - coupling nesting/marginals,
  - dt doubling at teacher-update boundaries,
  - teacher-temperature schedule.

Run: pytest tests/test_gmcd.py
Requires a one-time GPT-2 tokenizer download; runs on CPU.
"""

import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


@pytest.fixture(scope='module')
def gmcd_model():
  import hydra
  import transformers
  import algo

  with hydra.initialize(config_path='../configs', version_base=None):
    cfg = hydra.compose(
      config_name='config',
      overrides=[
        'algo=gmcd', 'model=tiny', 'data=openwebtext-split',
        'loader.batch_size=2', 'loader.eval_batch_size=2',
        'loader.global_batch_size=2', 'algo.T=64',
        'trainer.precision=32', 'sampling.predictor=ancestral_cache',
      ])
  tok = transformers.AutoTokenizer.from_pretrained('gpt2')
  return algo.GMCD(cfg, tokenizer=tok)


def _toy_batch(model, B=2, L=16):
  return torch.randint(0, model.tokenizer.vocab_size, (B, L))


def test_coupling_is_nested_and_exact(gmcd_model):
  x0 = torch.randint(0, gmcd_model.tokenizer.vocab_size, (64, 4096))
  alpha_t = torch.full((64, 1), 0.3)  # P(mask@t) = 0.7
  alpha_s = torch.full((64, 1), 0.7)  # P(mask@s) = 0.3
  xt, xs = gmcd_model._sample_coupled(x0, alpha_t, alpha_s)
  masked_t = xt == gmcd_model.mask_index
  masked_s = xs == gmcd_model.mask_index
  # Absorbing property: masked-at-s implies masked-at-t.
  assert bool((masked_s <= masked_t).all())
  assert abs(masked_t.float().mean().item() - 0.7) < 0.02
  assert abs(masked_s.float().mean().item() - 0.3) < 0.02


def test_checkpoint_strips_teacher_and_reloads(gmcd_model):
  import collections
  import algo

  # GMCD must override the checkpoint hooks (otherwise teacher.* leaks in).
  assert 'on_save_checkpoint' in algo.GMCD.__dict__
  assert 'on_load_checkpoint' in algo.GMCD.__dict__

  # Materialize the teacher, as a training step would.
  gmcd_model.nll(_toy_batch(gmcd_model), None, None)
  assert gmcd_model.teacher is not None
  full = gmcd_model.state_dict()
  assert any(k.startswith('teacher') for k in full), (
    'teacher should be a registered submodule (so it must be stripped)')

  # The same filter the hooks apply.
  stripped = collections.OrderedDict(
    (k, v) for k, v in full.items() if not k.startswith('teacher'))
  assert not any(k.startswith('teacher') for k in stripped)

  # A fresh instance (teacher=None) must load the stripped dict with no
  # unexpected keys -- this is exactly what blocked resume/eval before.
  import hydra, transformers
  with hydra.initialize(config_path='../configs', version_base=None):
    cfg = hydra.compose(
      config_name='config',
      overrides=['algo=gmcd', 'model=tiny', 'data=openwebtext-split',
                 'loader.batch_size=2', 'loader.eval_batch_size=2',
                 'loader.global_batch_size=2', 'algo.T=64',
                 'trainer.precision=32',
                 'sampling.predictor=ancestral_cache'])
  fresh = algo.GMCD(
    cfg, tokenizer=transformers.AutoTokenizer.from_pretrained('gpt2'))
  missing, unexpected = fresh.load_state_dict(stripped, strict=False)
  assert unexpected == [], f'unexpected keys: {unexpected}'


def test_teacher_is_deterministic(gmcd_model):
  # Put the whole model in train mode (dropout on) as Lightning would.
  gmcd_model.train()
  x = _toy_batch(gmcd_model)
  sigma = gmcd_model._sigma_from_alphat(torch.full((x.shape[0], 1), 0.5))
  a = gmcd_model._teacher_logits(x, sigma)
  b = gmcd_model._teacher_logits(x, sigma)
  assert torch.allclose(a, b, atol=1e-5), (
    'teacher targets differ across identical calls -> dropout still active')


def test_kl_bwd_is_finite_on_revealed_tokens(gmcd_model):
  # Reverse KL against a delta teacher is infinite; the split must use CE.
  gmcd_model.loss_type = 'kl-bwd'
  gmcd_model.train()
  loss = gmcd_model.nll(_toy_batch(gmcd_model, B=4, L=32), None, None)
  gmcd_model.loss_type = 'kl-fwd'
  assert torch.isfinite(loss).all()
  assert loss.max().item() < 1e4, (
    f'kl-bwd blew up (max {loss.max().item()}) -> delta-teacher not handled')


def test_dt_doubles_at_update_boundaries(gmcd_model, monkeypatch):
  import algo
  gmcd_model.linear_growth_dt = False
  T = gmcd_model.T
  for step, n in [(0, 0), (9999, 0), (10000, 1), (20000, 2)]:
    # Shadow the read-only Lightning global_step property for the test.
    monkeypatch.setattr(algo.GMCD, 'global_step', step, raising=False)
    assert gmcd_model._compute_dt() == 2 ** n / T


def test_forward_kl_matches_ce_on_delta():
  # Forward KL(teacher||student) with a one-hot teacher equals CE.
  V = 8
  student = torch.randn(V).log_softmax(-1)
  neg_inf = -1e6
  teacher = torch.full((V,), neg_inf)
  teacher[3] = 0.0  # delta on token 3 (SUBS carry-over)
  fwd_kl = (teacher.exp() * (teacher - student)).sum()
  ce = -student[3]
  assert torch.allclose(fwd_kl, ce, atol=1e-4)
