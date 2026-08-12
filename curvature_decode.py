"""E3: curvature-aware decoding for masked diffusion (the level-two flagship).

The claim under test: order parallel unmasking by *posterior curvature* rather
than by confidence. Curvature reads the continuous logit coordinate through the
second-order Tweedie structure, so it cannot be reduced to the first-order
prediction, which is what makes this the operation the canonical lift uniquely
enables.

Decoder. MaskGIT-style iterative unmasking in `k` steps. At each step the
denoiser predicts a categorical for every masked position; we commit the most
"decidable" positions and iterate. The methods differ only in the score used to
rank positions:

  confidence  commit highest max-prob first          (the standard rule)
  random      commit a random subset                 (control)
  entropy     commit lowest predictive entropy first  (first-order only)
  curvature   commit lowest posterior curvature first (second-order, ours)

Curvature. We feed the current state as continuous input logits (the soft-input
path of the DiT's embedding layer, softmax(l)@E) and estimate, per position, the
Frobenius sensitivity of the backbone output to random perturbations of the
still-masked context, tr(J^T J) via Hutchinson probes. A pinned token's
prediction is stable under context perturbation (low curvature -> commit first);
an undecided token's prediction swings with the uncertain context (high
curvature -> defer). Only the entropy baseline shares everything except this
second-order signal, so a win over entropy is attributable to the curvature.

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
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------- model
def build_model(args):
  """Return a masked-diffusion model exposing .backbone, .forward, .mask_index,
  .vocab_size, .noise, ._process_sigma, ._sigma_from_alphat, .tokenizer."""
  import hydra
  import transformers
  import algo

  overrides = [
    'algo=mdlm', 'data=openwebtext-split',
    'sampling.predictor=ancestral_cache',
    f'model={"tiny" if args.model == "toy" else "small"}',
    'loader.batch_size=1', 'loader.eval_batch_size=1',
    'loader.global_batch_size=1', 'trainer.precision=32',
  ]
  with hydra.initialize(config_path='configs', version_base=None):
    cfg = hydra.compose(config_name='config', overrides=overrides)
  tok = transformers.AutoTokenizer.from_pretrained('gpt2')
  if args.model == 'toy':
    model = algo.MDLM(cfg, tokenizer=tok)
  else:
    model = algo.MDLM.load_from_checkpoint(
      args.checkpoint, tokenizer=tok, config=cfg)
  model = model.to(args.device).eval()
  return model


# ------------------------------------------------------------- scoring rules
@torch.no_grad()
def denoise_logp(model, z, sigma):
  """Per-position log-probabilities (B, L, V), mask column excluded."""
  return model.forward(z, sigma=sigma)


def confidence_score(logp):
  # higher max log-prob -> more decidable -> commit first
  return logp.max(dim=-1).values


def neg_entropy_score(logp):
  # -entropy; higher (less negative) -> lower entropy -> commit first
  p = logp.exp()
  return (p * logp).sum(dim=-1)


@torch.no_grad()
def curvature_score(model, z, sigma, masked, n_probes, eps, in_scale):
  """Negative Hutchinson Frobenius sensitivity, so commit-first == low curvature.

  Feeds continuous input logits (soft-input embedding path), perturbs the masked
  context, and averages the squared change in each position's predicted
  distribution. This is tr(J^T J) of the output-vs-input Jacobian restricted to
  masked inputs, an unbiased Hutchinson estimate up to the finite-difference eps.
  """
  V = model.vocab_size
  proc_sigma = model._process_sigma(sigma)
  base = in_scale * F.one_hot(z, V).to(torch.float32)  # (B, L, V), near one-hot
  with torch.amp.autocast('cuda', enabled=False):
    p0 = model.backbone(x=base, sigma=proc_sigma).softmax(dim=-1)
    sens = torch.zeros(z.shape, device=z.device, dtype=torch.float32)
    m = masked.unsqueeze(-1).to(base.dtype)  # (B, L, 1) perturb masked only
    for _ in range(n_probes):
      pert = base + eps * torch.randn_like(base) * m
      p = model.backbone(x=pert, sigma=proc_sigma).softmax(dim=-1)
      sens += ((p - p0) ** 2).sum(dim=-1) / (eps * eps)
  return -(sens / n_probes)


ORDER_FNS = {'confidence', 'entropy', 'curvature', 'random'}


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
def decode(model, order, n_steps, n_samples, L, args, gen):
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
      if order == 'confidence':
        score = confidence_score(logp)
      elif order == 'entropy':
        score = neg_entropy_score(logp)
      elif order == 'curvature':
        score = curvature_score(model, z, sigma, masked, args.n_probes,
                                args.eps, args.in_scale)
        nfe += args.n_probes + 1
      elif order == 'random':
        score = torch.rand(z.shape, device=device)
      else:
        raise ValueError(order)
      score = score.masked_fill(~masked, -math.inf)  # only rank masked
      k = min(int(reveal[step]), n_masked)
      if k <= 0:
        continue
      idx = score.topk(k, dim=1).indices  # positions to commit
      fill = logp[..., :mask_id].argmax(dim=-1)  # argmax token (mask excluded)
      z = z.scatter(1, idx, fill.gather(1, idx))
    # fill any leftover masked positions
    left = z == mask_id
    if left.any():
      logp = denoise_logp(model, z,
                          model._sigma_from_alphat(torch.full((z.shape[0], 1),
                                                              1e-3, device=device)))
      nfe += 1
      z = torch.where(left, logp[..., :mask_id].argmax(dim=-1), z)
    out.append(z.cpu())
  return torch.cat(out), nfe, time.perf_counter() - t0


# ------------------------------------------------------------------ metrics
def distinct_n(tokens, n):
  grams, total = set(), 0
  for row in tokens.tolist():
    for i in range(len(row) - n + 1):
      grams.add(tuple(row[i:i + n]))
      total += 1
  return len(grams) / max(total, 1)


@torch.no_grad()
def gen_ppl(tokens, judge_name, device, batch_size=8):
  from transformers import AutoModelForCausalLM
  judge = AutoModelForCausalLM.from_pretrained(judge_name).to(device).eval()
  nll, count = 0.0, 0
  for i in range(0, tokens.shape[0], batch_size):
    ids = tokens[i:i + batch_size].to(device)
    nll += float(judge(ids, labels=ids).loss) * (ids.numel() - ids.shape[0])
    count += ids.numel() - ids.shape[0]
  return float(np.exp(nll / max(count, 1)))


# --------------------------------------------------------------------- main
def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--model', choices=['toy', 'ckpt'], default='ckpt')
  p.add_argument('--checkpoint', default=None)
  p.add_argument('--orders', nargs='+', default=['confidence', 'entropy',
                                                 'curvature', 'random'])
  p.add_argument('--steps', type=int, nargs='+', default=[2, 4, 8, 16, 32])
  p.add_argument('--n-samples', type=int, default=512)
  p.add_argument('--batch-size', type=int, default=64)
  p.add_argument('--seq-len', type=int, default=1024)
  p.add_argument('--n-probes', type=int, default=4, help='Hutchinson probes')
  p.add_argument('--eps', type=float, default=0.1, help='finite-diff step')
  p.add_argument('--in-scale', type=float, default=6.0,
                 help='input-logit sharpness. MUST be calibrated on GPU: too '
                      'high saturates softmax and the sensitivity collapses to '
                      '~0, too low blurs the state. Check that curvature scores '
                      'have spread across positions (not all near-equal).')
  p.add_argument('--judge', default='gpt2-large')
  p.add_argument('--no-ppl', action='store_true')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--output', default='curvature_decode_results.json')
  args = p.parse_args()
  args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if args.model == 'ckpt' and not args.checkpoint:
    p.error('--checkpoint required unless --model toy')
  if args.model == 'toy':
    args.seq_len = 32

  torch.manual_seed(args.seed)
  gen = torch.Generator(device='cpu').manual_seed(args.seed)
  model = build_model(args)
  L = min(args.seq_len, model.seq_len if hasattr(model, 'seq_len') else args.seq_len)

  results = []
  for order in args.orders:
    for n_steps in args.steps:
      tokens, nfe, secs = decode(model, order, n_steps, args.n_samples, L,
                                 args, gen)
      row = {
        'order': order, 'n_steps': n_steps, 'nfe_per_pass': nfe / max(
          1, math.ceil(args.n_samples / args.batch_size)),
        'distinct2': distinct_n(tokens, 2),
        'wall_seconds': secs,
      }
      if not args.no_ppl:
        row['gen_ppl'] = gen_ppl(tokens, args.judge, args.device)
      results.append(row)
      print(f'[e3] {row}', flush=True)

  with open(args.output, 'w') as f:
    json.dump(results, f, indent=2)
  print(f'[e3] wrote {args.output}')


if __name__ == '__main__':
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  main()
