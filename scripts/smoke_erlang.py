"""CPU smoke test for Erlang-k semi-Markov masking (algo=erlang_mdlm).

Verifies, on a tiny model with the GPT-2 tokenizer:
  1. Phase marginal: P(fully masked, j=k) matches 1 - alpha_t.
  2. Endpoint equivalence: the graded [token, mask] embedding at reveal in
     {0, 1} reproduces the hard mask / token embedding, so inference on hard
     states is in-distribution.
  3. k=1 reduction: on masked positions the graded head reproduces the
     inherited MDLM hard-state log-probabilities (same backbone weights).
  4. Training path: nll() is finite and backward() flows for k in {1, 2, 4}.

Run from the repo root: .venv/bin/python scripts/smoke_erlang.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
import transformers

import algo


def build(k):
  with hydra.initialize(config_path='../configs', version_base=None):
    config = hydra.compose(
      config_name='config',
      overrides=[
        'algo=erlang',
        'model=tiny',
        'data=openwebtext-split',
        'loader.batch_size=2',
        'loader.eval_batch_size=2',
        'loader.global_batch_size=2',
        f'algo.erlang_k={k}',
        'trainer.precision=32',
        'sampling.predictor=ancestral_cache',
      ])
  tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
  model = algo.ErlangMDLM(config, tokenizer=tokenizer)
  model.train()
  return model, tokenizer


def main():
  torch.manual_seed(0)
  B, L = 8, 128

  # (1) phase marginal
  model, tok = build(k=4)
  x0 = torch.randint(0, tok.vocab_size, (256, 512))
  for a in (0.2, 0.5, 0.8):
    alpha_t = torch.full((256, 1), a)
    phase = model._sample_phase(x0, alpha_t)
    frac_full = float((phase == model.erlang_k).float().mean())
    assert abs(frac_full - (1 - a)) < 0.01, (a, frac_full)
  print('[smoke] phase marginal ok: P(j=k) matches 1 - alpha_t within 0.01')

  # (2) endpoint embedding equivalence (reveal 0 -> mask, reveal 1 -> token)
  x0s = torch.randint(0, tok.vocab_size, (B, L))
  emb = model.backbone.vocab_embed
  idx = torch.stack([x0s, torch.full_like(x0s, model.mask_index)], dim=-1)
  w0 = torch.stack([torch.zeros_like(x0s).float(),
                    torch.ones_like(x0s).float()], dim=-1)
  w1 = torch.stack([torch.ones_like(x0s).float(),
                    torch.zeros_like(x0s).float()], dim=-1)
  graded_mask = emb(idx, w0)
  graded_tok = emb(idx, w1)
  hard_mask = emb(torch.full_like(x0s, model.mask_index))
  hard_tok = emb(x0s)
  assert torch.allclose(graded_mask, hard_mask, atol=1e-5)
  assert torch.allclose(graded_tok, hard_tok, atol=1e-5)
  print('[smoke] endpoint embeddings ok: graded r=0 == [MASK], r=1 == token')

  # (3) k=1 reduction: masked-position graded log-probs == hard-state log-probs
  model1, tok1 = build(k=1)
  x0b = torch.randint(0, tok1.vocab_size, (B, L))
  masked = torch.rand(B, L) < 0.5
  phase = masked.long()  # k=1: phase in {0,1}
  sigma = model1._sigma_from_alphat(torch.full((B, 1), 0.5))
  graded, _ = model1._graded_logits(x0b, phase, sigma, None)
  xt_hard = torch.where(masked, torch.full_like(x0b, model1.mask_index), x0b)
  hard = model1.forward(xt_hard, sigma=sigma)
  gp = torch.gather(graded, -1, x0b[:, :, None]).squeeze(-1)
  hp = torch.gather(hard, -1, x0b[:, :, None]).squeeze(-1)
  diff = (gp - hp)[masked].abs().max()
  assert float(diff) < 1e-4, float(diff)
  print(f'[smoke] k=1 reduction ok: masked log-prob max|diff| {float(diff):.2e}')

  # (4) training path for k in {1, 2, 4}
  for k in (1, 2, 4):
    m, tk = build(k=k)
    xb = torch.randint(0, tk.vocab_size, (B, L))
    loss_pt = m.nll(xb, None, None, current_accumulation_step=None)
    assert loss_pt.shape == (B, L)
    assert torch.isfinite(loss_pt).all(), f'non-finite loss at k={k}'
    loss = loss_pt.sum() / (loss_pt != 0).sum().clamp(min=1)
    loss.backward()
    gnorm = sum(p.grad.norm() ** 2 for p in m.backbone.parameters()
                if p.grad is not None) ** 0.5
    assert torch.isfinite(torch.tensor(float(gnorm))) and float(gnorm) > 0
    print(f'[smoke] k={k}: loss {float(loss):.4f}, grad norm '
          f'{float(gnorm):.4f} -- backward ok')

  print('[smoke] PASS')


if __name__ == '__main__':
  main()
