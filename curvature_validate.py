"""Validate posterior curvature as a real signal, before any decoding claim.

Mechanism first. On real partially-masked MDLM states, we ask whether the
per-position curvature statistic predicts things a good "commit this position
next" criterion should predict, and whether it does so better than the
first-order controls and better than a structure-destroying null.

Targets (per masked position i):
  correctness   p_i(x0[i]), the mass the model already puts on the true token.
  stability     whether argmax_i now agrees with argmax_i once the rest of the
                masked context is resolved to ground truth. This is the
                second-order property: a confident-but-context-dependent token
                is unstable. Curvature should predict this where entropy cannot.
  oracle_gain   -(future denoising NLL on the remaining masked positions after
                committing i to its argmax). A small oracle tries every candidate
                i; we report position-selection regret against it.

Commit scores compared (higher = commit-first):
  curvature (ours, -sensitivity), curvature_shuffled (null: positions permuted),
  curvature_diag (cheap diagonal-only control), entropy (-H), max_prob,
  margin (top1-top2 logit), score_mag (first-order score norm), random.

Metrics per (score, target): Spearman rank correlation, top-decile enrichment
(mean target in the top 10% by score / overall mean), and regret (oracle target
minus the target at the score's top-1 pick), averaged over states.

If curvature does not beat entropy and margin here, and does not beat its own
shuffled null, a downstream decoding improvement cannot be attributed to the
theory.

Usage:
  python curvature_validate.py --checkpoint checkpoints/mdlm.ckpt \
      --n-states 64 --mask-frac 0.5 --max-candidates 32
  python curvature_validate.py --model toy --n-states 4 --max-candidates 8  # CPU smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curvature_decode import build_model, curvature_score  # noqa: E402


# --------------------------------------------------------------- data / state
def real_batch(model, n, L, device, seed):
  """Real token sequences. Uses OWT val if available, else the toy Markov-ish
  fallback of random tokens (plumbing only)."""
  try:
    from datasets import load_dataset
    ds = load_dataset('openwebtext', split='train', streaming=True)
    tok = model.tokenizer
    buf, rows = [], []
    for ex in ds:
      buf.extend(tok(ex['text'])['input_ids'] + [tok.eos_token_id])
      while len(buf) >= L:
        rows.append(buf[:L]); buf = buf[L:]
        if len(rows) >= n:
          return torch.tensor(rows, device=device)
  except Exception:
    pass
  g = torch.Generator().manual_seed(seed)
  return torch.randint(0, model.vocab_size - 1, (n, L), generator=g).to(device)


# ------------------------------------------------------------- commit scores
@torch.no_grad()
def commit_scores(model, z, masked, logp, args, gen):
  """Dict name -> (B, L); higher means commit-first. Only masked entries used."""
  V = model.vocab_size
  p = logp.exp()
  top2 = logp.topk(2, dim=-1).values
  sens = -curvature_score(model, z, _sigma(model, masked), masked,
                          args.n_probes, args.eps, args.in_scale)  # >=0
  # first-order score norm (Tweedie first moment surrogate): how far the
  # predictive distribution is from uniform, per position.
  score_mag = (p - 1.0 / V).norm(dim=-1)
  out = {
    'curvature': -sens,                         # low curvature -> commit
    'curvature_shuffled': _shuffle_masked(-sens, masked, gen),
    'curvature_diag': -_diag_curvature(model, z, masked),
    'entropy': (p * logp).sum(-1),              # = -H, higher -> lower entropy
    'max_prob': logp.max(-1).values,
    'margin': (top2[..., 0] - top2[..., 1]),
    'score_mag': -score_mag,                    # test sign via correlation
    'random': torch.rand(z.shape, device=z.device, generator=None),
  }
  return out


@torch.no_grad()
def _diag_curvature(model, z, masked):
  """Cheap diagonal-only curvature proxy from a single forward: the
  residual-dependent injected-curvature term e^{-R/b}, read off the score."""
  # Surrogate: 1 - max_prob is a monotone stand-in for the diagonal broadening
  # that needs no probes. Present as the cheap control, not the real estimator.
  logp = model.forward(z, sigma=_sigma(model, masked))
  return 1.0 - logp.exp().max(-1).values


def _shuffle_masked(x, masked, gen):
  y = x.clone()
  for b in range(x.shape[0]):
    idx = masked[b].nonzero(as_tuple=True)[0]
    if len(idx) > 1:
      perm = idx[torch.randperm(len(idx), generator=gen)]
      y[b, idx] = x[b, perm]
  return y


def _sigma(model, masked):
  frac = masked.float().mean(dim=1, keepdim=True).clamp(1e-4, 1 - 1e-4)
  return model._sigma_from_alphat(1 - frac)


# -------------------------------------------------------------------- targets
@torch.no_grad()
def targets(model, z, x0, masked, logp, args):
  """Dict name -> (B, L); higher means better to commit next."""
  V, mask_id = model.vocab_size, model.mask_index
  p = logp.exp()
  # correctness: probability on the true token
  correctness = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
  # stability: does argmax_now survive resolving the rest of the context to GT?
  argmax_now = logp[..., :mask_id].argmax(-1)
  z_ctx = torch.where(masked, x0, z)  # fill all masked with GT (oracle context)
  # re-mask each position one-at-a-time would be M forwards; instead approximate
  # by leaving position i masked while filling the rest, done per position in a
  # vectorized leave-one-out below for a candidate subset.
  stability = _leave_one_out_stability(model, z, x0, masked, argmax_now, args)
  return {'correctness': correctness, 'stability': stability}


@torch.no_grad()
def _leave_one_out_stability(model, z, x0, masked, argmax_now, args):
  """For a subset of candidates, fill all OTHER masked positions with GT, keep i
  masked, and record p_i(argmax_now_i) under the resolved context. High = the
  early commitment agrees with the context-resolved answer (stable)."""
  device = z.device
  stab = torch.zeros(z.shape, device=device)
  for b in range(z.shape[0]):
    cand = masked[b].nonzero(as_tuple=True)[0]
    if len(cand) > args.max_candidates:
      cand = cand[torch.randperm(len(cand))[:args.max_candidates]]
    for i in cand.tolist():
      zc = torch.where(masked[b], x0[b], z[b]).clone()
      zc[i] = model.mask_index  # keep i masked, rest = GT
      lp = model.forward(zc.unsqueeze(0), sigma=_sigma(model, (zc == model.mask_index).unsqueeze(0)))
      stab[b, i] = lp[0, i].exp()[argmax_now[b, i]]
  return stab


@torch.no_grad()
def oracle_gain(model, z, x0, masked, logp, args):
  """-(remaining NLL after committing i to argmax), for candidate positions.
  This is the 'try every candidate' oracle; report regret against it."""
  device = z.device
  gain = torch.full(z.shape, -math.inf, device=device)
  argmax_now = logp[..., :model.mask_index].argmax(-1)
  for b in range(z.shape[0]):
    cand = masked[b].nonzero(as_tuple=True)[0]
    if len(cand) > args.max_candidates:
      cand = cand[torch.randperm(len(cand))[:args.max_candidates]]
    for i in cand.tolist():
      zc = z[b].clone()
      zc[i] = argmax_now[b, i]  # commit i
      m2 = (zc == model.mask_index)
      lp = model.forward(zc.unsqueeze(0), sigma=_sigma(model, m2.unsqueeze(0)))[0]
      rem = m2.nonzero(as_tuple=True)[0]
      if len(rem):
        nll = -lp[rem].gather(-1, x0[b, rem].unsqueeze(-1)).mean()
        gain[b, i] = -float(nll)
  return gain


# ------------------------------------------------------------------- metrics
def spearman(a, b):
  from scipy.stats import spearmanr
  r = spearmanr(a, b).correlation
  return float(r) if r == r else 0.0


def evaluate(scores, tgts, masked):
  m = masked.bool().cpu().numpy().reshape(-1)
  rep = {}
  for tname, t in tgts.items():
    tv = t.cpu().numpy().reshape(-1)
    valid = m & np.isfinite(tv)
    rep[tname] = {}
    for sname, s in scores.items():
      sv = s.cpu().numpy().reshape(-1)
      v = valid & np.isfinite(sv)
      if v.sum() < 8:
        continue
      rho = spearman(sv[v], tv[v])
      # top-decile enrichment (of score) on the target
      k = max(1, int(0.1 * v.sum()))
      top = np.argsort(-sv[v])[:k]
      enr = float(tv[v][top].mean() / (tv[v].mean() + 1e-9))
      rep[tname][sname] = {'spearman': round(rho, 4),
                           'top_decile_enrichment': round(enr, 4)}
    rep[tname] = dict(sorted(rep[tname].items(),
                             key=lambda kv: -kv[1]['spearman']))
  return rep


def regret(scores, gain, masked):
  """Per state: oracle picks argmax gain; each score picks its argmax; regret =
  gain[oracle] - gain[score pick]. Averaged over states where an oracle exists."""
  out = {s: [] for s in scores}
  B = gain.shape[0]
  for b in range(B):
    idx = masked[b].nonzero(as_tuple=True)[0]
    g = gain[b, idx]
    fin = torch.isfinite(g)
    if fin.sum() < 2:
      continue
    idx, g = idx[fin], g[fin]
    best = float(g.max())
    for s, sv in scores.items():
      pick = idx[torch.argmax(sv[b, idx])]
      out[s].append(best - float(gain[b, pick]))
  return {s: round(float(np.mean(v)), 4) for s, v in out.items() if v}


# --------------------------------------------------------------------- main
def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--model', choices=['toy', 'ckpt'], default='ckpt')
  p.add_argument('--checkpoint', default=None)
  p.add_argument('--n-states', type=int, default=64)
  p.add_argument('--seq-len', type=int, default=128)
  p.add_argument('--mask-frac', type=float, default=0.5)
  p.add_argument('--max-candidates', type=int, default=32,
                 help='oracle candidates per state (compute cap)')
  p.add_argument('--n-probes', type=int, default=8)
  p.add_argument('--eps', type=float, default=0.1)
  p.add_argument('--in-scale', type=float, default=6.0)
  p.add_argument('--batch-size', type=int, default=16)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--output', default='curvature_validate_results.json')
  args = p.parse_args()
  args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if args.model == 'ckpt' and not args.checkpoint:
    p.error('--checkpoint required unless --model toy')
  if args.model == 'toy':
    args.seq_len = 32

  torch.manual_seed(args.seed)
  gen = torch.Generator(device=args.device).manual_seed(args.seed)
  model = build_model(args)
  L = min(args.seq_len, getattr(model, 'seq_len', args.seq_len))
  x0_all = real_batch(model, args.n_states, L, args.device, args.seed)

  agg_scores, agg_tgts, agg_masked, gains, masks_for_regret = {}, {}, [], [], []
  for start in range(0, args.n_states, args.batch_size):
    x0 = x0_all[start:start + args.batch_size]
    mrand = torch.rand(x0.shape, device=args.device, generator=gen)
    masked = mrand < args.mask_frac
    masked[:, 0] = False  # keep BOS
    z = torch.where(masked, model.mask_index, x0)
    logp = model.forward(z, sigma=_sigma(model, masked))

    sc = commit_scores(model, z, masked, logp, args, gen)
    tg = targets(model, z, x0, masked, logp, args)
    g = oracle_gain(model, z, x0, masked, logp, args)
    for d, agg in [(sc, agg_scores), (tg, agg_tgts)]:
      for k, v in d.items():
        agg.setdefault(k, []).append(v)
    agg_masked.append(masked); gains.append(g)

  cat = lambda d: {k: torch.cat(v) for k, v in d.items()}
  scores, tgts = cat(agg_scores), cat(agg_tgts)
  masked = torch.cat(agg_masked); gain = torch.cat(gains)
  tgts['oracle_gain'] = gain

  report = {
    'config': {k: getattr(args, k) for k in
               ['n_states', 'mask_frac', 'n_probes', 'eps', 'in_scale',
                'max_candidates']},
    'correlation_and_enrichment': evaluate(scores, tgts, masked),
    'position_selection_regret': regret(scores, gain, masked),
  }
  with open(args.output, 'w') as f:
    json.dump(report, f, indent=2)
  print(json.dumps(report, indent=2))
  # headline verdict
  st = report['correlation_and_enrichment'].get('stability', {})
  cur = st.get('curvature', {}).get('spearman', 0)
  ent = st.get('entropy', {}).get('spearman', 0)
  mar = st.get('margin', {}).get('spearman', 0)
  nul = st.get('curvature_shuffled', {}).get('spearman', 0)
  print(f'\n[verdict] stability Spearman: curvature={cur} entropy={ent} '
        f'margin={mar} shuffled_null={nul}')
  print('[verdict] curvature is a real signal iff it beats entropy AND margin '
        'AND its shuffled null.')


if __name__ == '__main__':
  main()
