"""CPU smoke test for the G-MCD algorithm (algo=gmcd).

Composes the hydra config with a tiny model, instantiates GMCD with the GPT-2
tokenizer, and runs one coupled-trajectory distillation loss with a backward
pass. Verifies the threshold coupling invariants on the way.

Run from the repo root: .venv/bin/python scripts/smoke_gmcd.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
import transformers

import algo


def main():
  with hydra.initialize(config_path='../configs', version_base=None):
    config = hydra.compose(
      config_name='config',
      overrides=[
        'algo=gmcd',
        'model=tiny',
        'data=openwebtext-split',
        'loader.batch_size=2',
        'loader.eval_batch_size=2',
        'loader.global_batch_size=2',
        'algo.T=64',
        'trainer.precision=32',
        'sampling.predictor=ancestral_cache',
      ])
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  model = algo.GMCD(config, tokenizer=tokenizer)
  model.train()

  B, L = 2, 64
  x0 = torch.randint(0, tokenizer.vocab_size, (B, L))

  # Coupling invariants: nested mask sets, exact marginals at scale.
  alpha_t = torch.full((B, 1), 0.3)
  alpha_s = torch.full((B, 1), 0.7)
  xt, xs = model._sample_coupled(x0, alpha_t, alpha_s)
  masked_t = xt == model.mask_index
  masked_s = xs == model.mask_index
  assert bool((masked_s <= masked_t).all()), 'mask sets must be nested'
  big = torch.randint(0, tokenizer.vocab_size, (64, 4096))
  bt, bs = model._sample_coupled(
    big, torch.full((64, 1), 0.3), torch.full((64, 1), 0.7))
  frac_t = float((bt == model.mask_index).float().mean())
  frac_s = float((bs == model.mask_index).float().mean())
  assert abs(frac_t - 0.7) < 0.01 and abs(frac_s - 0.3) < 0.01, (
    frac_t, frac_s)
  print(f'[smoke] coupling ok: P(mask@t)={frac_t:.4f} (0.7), '
        f'P(mask@s)={frac_s:.4f} (0.3), nested sets verified')

  # One distillation loss + backward.
  loss_per_token = model.nll(x0, None, None,
                             current_accumulation_step=None)
  assert loss_per_token.shape == (B, L)
  assert torch.isfinite(loss_per_token).all(), 'loss has non-finite entries'
  loss = loss_per_token.mean()
  loss.backward()
  grad_norm = sum(
    p.grad.norm() ** 2 for p in model.backbone.parameters()
    if p.grad is not None) ** 0.5
  assert torch.isfinite(torch.tensor(float(grad_norm)))
  print(f'[smoke] gmcd loss {float(loss):.4f}, grad norm '
        f'{float(grad_norm):.4f} -- backward ok')

  # Teacher must be populated and frozen relative to autograd.
  assert model.teacher is not None
  assert all(not p.requires_grad or p.grad is None
             for p in model.teacher.parameters())
  print('[smoke] teacher constructed lazily, no grads flow into it')
  print('[smoke] PASS')


if __name__ == '__main__':
  main()
