"""Validate exact Gumbel-Tweedie curvature before any decoding claim.

Mechanism first. On real partially-masked MDLM states, we ask whether the
per-position posterior-score covariance predicts things a good "commit this
position next" criterion should predict, and whether it does so better than
first-order controls and a structure-destroying null.

Targets (per masked position i):
  correctness   p_i(x0[i]), the mass the model already puts on the true token.
  stability     whether argmax_i now agrees with argmax_i once the rest of the
                masked context is resolved to ground truth. A confident token
                can still be context-dependent and unstable; the sensitivity
                proxy should expose that distinction.
  oracle_gain   -(future denoising NLL on the remaining masked positions after
                committing i to its argmax). A small oracle tries every candidate
                i; we report position-selection regret against it.

Commit scores compared (higher = commit-first):
  curvature (negative exact Gumbel-Tweedie covariance trace),
  curvature_shuffled (null: positions permuted), Gini (an algebraically
  equivalent control on the neutral lift), entropy (-H), max_prob, margin
  (top1-top2 logit), distance_from_uniform, sensitivity (the former
  finite-difference proxy), and random.

Metrics per (score, target): Spearman rank correlation, top-decile mean/lift,
nonnegative-target enrichment, and regret (oracle target minus the target at
the score's top-1 pick), averaged over states.

If curvature does not beat entropy, margin, and its shuffled null here, it is
not a useful mechanism for downstream decoding. Because the neutral-lift exact
trace is a fixed multiple of Gini, this experiment tests Gini-based allocation;
it cannot by itself establish a uniquely Gumbel-specific empirical advantage.
The conditional-moment formula is exact, while substituting the discrete MDLM
posterior for a continuous-lift posterior is an explicit modeling assumption.

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
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curvature_decode import (  # noqa: E402
    build_model,
    curvature_score,
    sensitivity_score,
)


# --------------------------------------------------------------- data / state
def real_batch(
    model, n, L, device, seed, data_source="openwebtext", allow_random_fallback=False
):
    """Returns token sequences and a provenance label for the selected source."""
    if data_source == "random":
        return _random_batch(model, n, L, device, seed), "random"
    if data_source != "openwebtext":
        raise ValueError(f"unknown data source: {data_source}")
    failure = None
    try:
        from datasets import load_dataset

        ds = load_dataset("openwebtext", split="train", streaming=True)
        tok = model.tokenizer
        buf, rows = [], []
        for ex in ds:
            buf.extend(tok(ex["text"])["input_ids"] + [tok.eos_token_id])
            while len(buf) >= L:
                rows.append(buf[:L])
                buf = buf[L:]
                if len(rows) >= n:
                    print(f"[data] using real OpenWebText ({n} sequences)", flush=True)
                    return torch.tensor(rows, device=device), "openwebtext"
        failure = RuntimeError(
            f"OpenWebText ended after {len(rows)} sequences; requested {n}"
        )
    except Exception as e:
        failure = e
    if not allow_random_fallback:
        raise RuntimeError(
            "Could not load enough real OpenWebText sequences. Refusing to report "
            "scientific validation metrics on random tokens; fix dataset access or "
            "pass --allow-random-fallback for a plumbing-only run."
        ) from failure
    print(f"[data] OWT load failed ({type(failure).__name__}: {failure})", flush=True)
    print(
        "[data] WARNING: falling back to RANDOM tokens. Results are NOT a real "
        "validation (model predictions on non-language are noise). Fix data "
        "access before trusting any numbers.",
        flush=True,
    )
    return _random_batch(model, n, L, device, seed), "random_fallback"


def _random_batch(model, n, L, device, seed):
    """Creates deterministic plumbing-only tokens without assuming mask position."""
    g = torch.Generator().manual_seed(seed)
    clean_ids = torch.arange(model.vocab_size)
    clean_ids = clean_ids[clean_ids != model.mask_index]
    sampled = torch.randint(0, len(clean_ids), (n, L), generator=g)
    return clean_ids[sampled].to(device)


def candidate_mask(masked, max_candidates, generator):
    """Selects one reproducible candidate subset per state."""
    selected = torch.zeros_like(masked, dtype=torch.bool)
    for batch_index in range(masked.shape[0]):
        indices = masked[batch_index].nonzero(as_tuple=True)[0]
        if len(indices) > max_candidates:
            permutation = torch.randperm(
                len(indices), device=indices.device, generator=generator
            )
            indices = indices[permutation[:max_candidates]]
        selected[batch_index, indices] = True
    return selected


# ------------------------------------------------------------- commit scores
@torch.no_grad()
def commit_scores(model, z, masked, logp, args, gen):
    """Dict name -> (B, L); higher means commit-first. Only masked entries used."""
    logp = logp.float()
    p = logp.exp()
    clean_logp = logp.clone()
    clean_logp[..., model.mask_index] = -math.inf
    top2 = clean_logp.topk(2, dim=-1).values
    clean_p = torch.softmax(clean_logp, dim=-1)
    curvature = curvature_score(
        logp,
        model.mask_index,
        signal_scale=args.signal_scale,
        noise_scale=args.gumbel_scale,
    )
    sensitivity = sensitivity_score(
        model,
        z,
        _sigma(model, masked),
        masked,
        args.n_probes,
        args.eps,
        args.in_scale,
        gen,
    )
    gini = 1.0 - clean_p.square().sum(dim=-1)
    # First-order control: distance of the predictive categorical from uniform.
    uniform = torch.full_like(p, 1.0 / (model.vocab_size - 1))
    uniform[..., model.mask_index] = 0
    distance_from_uniform = (p - uniform).norm(dim=-1)
    safe_logp = torch.where(p > 0, logp, torch.zeros_like(logp))
    out = {
        "curvature": curvature,
        "curvature_shuffled": _shuffle_masked(curvature, masked, gen),
        "gini": -gini,
        "sensitivity": sensitivity,
        "sensitivity_shuffled": _shuffle_masked(sensitivity, masked, gen),
        "entropy": (p * safe_logp).sum(-1),  # = -H, higher -> lower entropy
        "max_prob": clean_logp.max(-1).values,
        "margin": (top2[..., 0] - top2[..., 1]),
        "distance_from_uniform": distance_from_uniform,  # farther = confident
        "random": torch.rand(z.shape, device=z.device, generator=gen),
    }
    return out


def _shuffle_masked(x, masked, gen):
    y = x.clone()
    for b in range(x.shape[0]):
        idx = masked[b].nonzero(as_tuple=True)[0]
        if len(idx) > 1:
            perm = idx[torch.randperm(len(idx), device=idx.device, generator=gen)]
            y[b, idx] = x[b, perm]
    return y


def _sigma(model, masked):
    frac = masked.float().mean(dim=1, keepdim=True).clamp(1e-4, 1 - 1e-4)
    return model._sigma_from_alphat(1 - frac)


# -------------------------------------------------------------------- targets
@torch.no_grad()
def initial_targets(model, x0, logp):
    """Returns correctness and the clean-token prediction before commitment."""
    p = logp.float().exp()
    # correctness: probability on the true token
    correctness = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    clean_logp = logp.float().clone()
    clean_logp[..., model.mask_index] = -math.inf
    argmax_now = clean_logp.argmax(-1)
    return correctness, argmax_now


@torch.no_grad()
def _leave_one_out_stability(model, z, x0, masked, argmax_now, selected_candidates):
    """Measures agreement and probability after resolving all other masks."""
    device = z.device
    agreement = torch.zeros(z.shape, device=device, dtype=torch.float32)
    probability = torch.zeros(z.shape, device=device, dtype=torch.float32)
    for b in range(z.shape[0]):
        cand = selected_candidates[b].nonzero(as_tuple=True)[0]
        for i in cand.tolist():
            zc = torch.where(masked[b], x0[b], z[b]).clone()
            zc[i] = model.mask_index  # keep i masked, rest = GT
            lp = model.forward(
                zc.unsqueeze(0),
                sigma=_sigma(model, (zc == model.mask_index).unsqueeze(0)),
            )
            resolved_logp = lp[0, i].float()
            resolved_logp[model.mask_index] = -math.inf
            early_token = argmax_now[b, i]
            probability[b, i] = resolved_logp.exp()[early_token]
            agreement[b, i] = (resolved_logp.argmax() == early_token).to(torch.float32)
    return {
        "stability": agreement,
        "stability_probability": probability,
    }


@torch.no_grad()
def oracle_gain(model, z, x0, masked, logp, selected_candidates):
    """-(remaining NLL after committing i to argmax), for candidate positions.
    This is the 'try every candidate' oracle; report regret against it."""
    device = z.device
    gain = torch.full(z.shape, -math.inf, device=device, dtype=torch.float32)
    clean_logp = logp.float().clone()
    clean_logp[..., model.mask_index] = -math.inf
    argmax_now = clean_logp.argmax(-1)
    for b in range(z.shape[0]):
        cand = selected_candidates[b].nonzero(as_tuple=True)[0]
        for i in cand.tolist():
            zc = z[b].clone()
            zc[i] = argmax_now[b, i]  # commit i
            m2 = zc == model.mask_index
            lp = model.forward(zc.unsqueeze(0), sigma=_sigma(model, m2.unsqueeze(0)))[0]
            rem = m2.nonzero(as_tuple=True)[0]
            if len(rem):
                nll = -lp[rem].float().gather(-1, x0[b, rem].unsqueeze(-1)).mean()
                gain[b, i] = -nll
    return gain


# ------------------------------------------------------------------- metrics
def spearman(a, b):
    from scipy.stats import spearmanr

    r = spearmanr(a, b).correlation
    return float(r) if r == r else 0.0


def _as_numpy(tensor):
    """Converts any floating PyTorch dtype, including bfloat16, to NumPy."""
    return tensor.detach().float().cpu().numpy()


def evaluate(scores, tgts, validity):
    rep = {}
    for tname, t in tgts.items():
        tv = _as_numpy(t).reshape(-1)
        valid = validity[tname].detach().bool().cpu().numpy().reshape(-1)
        valid &= np.isfinite(tv)
        rep[tname] = {}
        for sname, s in scores.items():
            sv = _as_numpy(s).reshape(-1)
            v = valid & np.isfinite(sv)
            if v.sum() < 8:
                continue
            rho = spearman(sv[v], tv[v])
            # top-decile enrichment (of score) on the target
            k = max(1, int(0.1 * v.sum()))
            top = np.argsort(-sv[v])[:k]
            overall_mean = float(tv[v].mean())
            top_mean = float(tv[v][top].mean())
            metrics = {
                "spearman": round(rho, 4),
                "top_decile_mean": round(top_mean, 4),
                "overall_mean": round(overall_mean, 4),
                "top_decile_lift": round(top_mean - overall_mean, 4),
            }
            # Ratios are interpretable only for nonnegative targets. In particular,
            # oracle_gain is negative NLL, for which the old ratio inverted meaning.
            if bool((tv[v] >= 0).all()) and overall_mean > 0:
                metrics["top_decile_enrichment"] = round(top_mean / overall_mean, 4)
            rep[tname][sname] = metrics
        rep[tname] = dict(sorted(rep[tname].items(), key=lambda kv: -kv[1]["spearman"]))
    return rep


def regret(scores, gain, selected_candidates):
    """Per state: oracle picks argmax gain; each score picks its argmax; regret =
    gain[oracle] - gain[score pick]. Averaged over states where an oracle exists."""
    out = {s: [] for s in scores}
    B = gain.shape[0]
    for b in range(B):
        idx = selected_candidates[b].nonzero(as_tuple=True)[0]
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
        "--allow-random-fallback",
        action="store_true",
        help="allow random-token plumbing data if OWT is unavailable",
    )
    p.add_argument(
        "--data-source",
        choices=["openwebtext", "random"],
        default=None,
        help="toy defaults to random; checkpoints to OWT",
    )
    p.add_argument("--n-states", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--mask-frac", type=float, default=0.5)
    p.add_argument(
        "--max-candidates",
        type=int,
        default=32,
        help="oracle candidates per state (compute cap)",
    )
    p.add_argument("--signal-scale", type=float, default=1.0)
    p.add_argument("--gumbel-scale", type=float, default=1.0)
    p.add_argument("--n-probes", type=int, default=8)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--in-scale", type=float, default=6.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="curvature_validate_results.json")
    args = p.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "ckpt" and not args.checkpoint:
        p.error("--checkpoint required unless --model toy")
    if args.model == "toy":
        args.seq_len = 32
        args.allow_random_fallback = True
    if args.data_source is None:
        args.data_source = "random" if args.model == "toy" else "openwebtext"
    if args.n_states <= 0 or args.seq_len <= 0 or args.batch_size <= 0:
        p.error("state, sequence, and batch sizes must be positive")
    if not 0 < args.mask_frac < 1:
        p.error("--mask-frac must lie strictly between zero and one")
    if args.max_candidates <= 0 or args.n_probes <= 0:
        p.error("--max-candidates and --n-probes must be positive")
    if args.signal_scale <= 0 or args.gumbel_scale <= 0:
        p.error("--signal-scale and --gumbel-scale must be positive")
    if args.eps <= 0 or args.in_scale <= 0:
        p.error("--eps and --in-scale must be positive")

    torch.manual_seed(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    model = build_model(args)
    model_length = getattr(getattr(model.config, "model", None), "length", args.seq_len)
    L = min(args.seq_len, model_length)
    x0_all, actual_data_source = real_batch(
        model,
        args.n_states,
        L,
        args.device,
        args.seed,
        data_source=args.data_source,
        allow_random_fallback=args.allow_random_fallback,
    )

    agg_scores, agg_tgts, agg_validity = {}, {}, {}
    agg_masked, agg_candidates, gains = [], [], []
    for start in range(0, args.n_states, args.batch_size):
        x0 = x0_all[start : start + args.batch_size]
        mrand = torch.rand(x0.shape, device=args.device, generator=gen)
        masked = mrand < args.mask_frac
        masked[:, 0] = False  # keep BOS
        selected_candidates = candidate_mask(masked, args.max_candidates, gen)
        z = torch.where(masked, model.mask_index, x0)
        logp = model.forward(z, sigma=_sigma(model, masked))

        sc = commit_scores(model, z, masked, logp, args, gen)
        correctness, argmax_now = initial_targets(model, x0, logp)
        stability = _leave_one_out_stability(
            model, z, x0, masked, argmax_now, selected_candidates
        )
        g = oracle_gain(model, z, x0, masked, logp, selected_candidates)
        tg = {"correctness": correctness, **stability, "oracle_gain": g}
        valid = {
            "correctness": masked,
            "stability": selected_candidates,
            "stability_probability": selected_candidates,
            "oracle_gain": selected_candidates & torch.isfinite(g),
        }
        for d, agg in [(sc, agg_scores), (tg, agg_tgts)]:
            for k, v in d.items():
                agg.setdefault(k, []).append(v)
        for k, v in valid.items():
            agg_validity.setdefault(k, []).append(v)
        agg_masked.append(masked)
        agg_candidates.append(selected_candidates)
        gains.append(g)

    cat = lambda d: {k: torch.cat(v) for k, v in d.items()}
    scores, tgts = cat(agg_scores), cat(agg_tgts)
    validity = cat(agg_validity)
    masked = torch.cat(agg_masked)
    selected_candidates = torch.cat(agg_candidates)
    gain = torch.cat(gains)

    report = {
        "config": {
            k: getattr(args, k)
            for k in [
                "model",
                "checkpoint",
                "architecture",
                "disable_ema",
                "n_states",
                "seq_len",
                "mask_frac",
                "signal_scale",
                "gumbel_scale",
                "n_probes",
                "eps",
                "in_scale",
                "max_candidates",
                "batch_size",
                "seed",
            ]
        },
        "data": {
            "requested_source": args.data_source,
            "actual_source": actual_data_source,
            "random_fallback_allowed": args.allow_random_fallback,
            "masked_positions": int(masked.sum()),
            "candidate_positions": int(selected_candidates.sum()),
        },
        "correlation_and_enrichment": evaluate(scores, tgts, validity),
        "position_selection_regret": regret(scores, gain, selected_candidates),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    # headline verdict
    st = report["correlation_and_enrichment"].get("stability", {})
    cur = st.get("curvature", {}).get("spearman", 0)
    ent = st.get("entropy", {}).get("spearman", 0)
    mar = st.get("margin", {}).get("spearman", 0)
    nul = st.get("curvature_shuffled", {}).get("spearman", 0)
    gin = st.get("gini", {}).get("spearman", 0)
    print(
        f"\n[verdict] stability Spearman: curvature={cur} entropy={ent} "
        f"margin={mar} gini={gin} shuffled_null={nul}"
    )
    print(
        "[verdict] exact neutral-lift Tweedie curvature is rank-equivalent to "
        "Gini; it is useful for decoding only if that shared ranking beats "
        "entropy, margin, and the shuffled null."
    )


if __name__ == "__main__":
    main()
