"""Token-level A_i characterization on a real MDLM.

Resolves the caveat from the char-level gate: on a real 50k-token vocabulary, how
many alternatives are actually plausible at a masked position, and how large must
the augmented state A_i = (top-m tokens, B-bit scores) be to retain the residual
signal? Uses the MDLM's own contextual posteriors p_theta(X_i | z), so the coupling
is real (not a char bigram).

Two reports:
  1. Effective plausible support of real token posteriors (participation ratio
     1/sum p^2, entropy, tokens for 90% mass). This is the decisive number: if the
     support is tens-to-hundreds, A_i is a genuine compression of the 50k vocab.
  2. A_i retention: lift the true token (v = a e_x + b G), compute the residual
     curvature with the MDLM posterior, a commitment_gain outcome via MDLM
     leave-one-out, and the top-m/B-bit retention within fixed-Gini bins.

Run (GPU): python characterize_Ai_token.py --checkpoint checkpoints/mdlm.ckpt \
             --n-seqs 64 --mask-frac 0.5 --max-candidates 16
     (CPU plumbing smoke): python characterize_Ai_token.py --model toy --n-seqs 4
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curvature_decode import build_model


# ---- analytic helpers (inlined so the fork is self-contained) ----
def gini(p): return 1 - (p ** 2).sum(-1)
def entropy(p): return -(np.where(p > 0, p * np.log(p), 0.0)).sum(-1)


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0: return 0.0
    r = spearmanr(x, y).correlation
    return float(r) if r == r else 0.0


def within_bin_spearman(score, target, gini_vals, n_bins=20):
    order = np.argsort(gini_vals)
    rs = [spearman(score[b], target[b]) for b in np.array_split(order, n_bins)
          if len(b) >= 16]
    return float(np.mean(rs)) if rs else 0.0


def curvature_full(v, p, a, b):
    C0 = np.expm1(a / b) ** 2 / b ** 2
    return C0 * (np.exp(-2 * v / b) * p * (1 - p)).sum(-1)


def curvature_from_Ai(v, p, a, b, m, B, v_lo=-3.0, v_hi=5.0):
    C0 = np.expm1(a / b) ** 2 / b ** 2
    w = np.exp(-2 * v / b)
    wbar = w.mean(-1, keepdims=True)
    idx = np.argsort(-p, axis=-1)[..., :m]
    levels = max(2, 2 ** B)
    vq = np.clip(v, v_lo, v_hi)
    vq = np.round((vq - v_lo) / (v_hi - v_lo) * (levels - 1)) / (levels - 1)
    vq = vq * (v_hi - v_lo) + v_lo
    wq = np.exp(-2 * vq / b)
    w_eff = np.broadcast_to(wbar, w.shape).copy()
    np.put_along_axis(w_eff, idx, np.take_along_axis(wq, idx, -1), axis=-1)
    return C0 * (w_eff * p * (1 - p)).sum(-1)


def effective_support(p):
    """Participation ratio, entropy, and tokens-for-90%-mass, per row."""
    pr = 1.0 / (p ** 2).sum(-1)
    H = entropy(p)
    ps = -np.sort(-p, axis=-1)
    cum = np.cumsum(ps, axis=-1)
    n90 = (cum < 0.9).sum(-1) + 1
    return pr, H, n90


@torch.no_grad()
def posteriors(model, z):
    """Marginal posteriors p_theta(X | z), mask column removed and renormalized."""
    logp = model.forward(z, sigma=_sigma(model, z == model.mask_index)).float()
    p = logp.exp()
    p[..., model.mask_index] = 0
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    return p


def _sigma(model, masked):
    frac = masked.float().mean(-1, keepdim=True).clamp(1e-4, 1 - 1e-4)
    return model._sigma_from_alphat(1 - frac)


def real_text(model, n, L, device, seed):
    try:
        from datasets import load_dataset
        ds = load_dataset("openwebtext", split="train", streaming=True)
        tok, buf, rows = model.tokenizer, [], []
        for ex in ds:
            buf += tok(ex["text"])["input_ids"] + [tok.eos_token_id]
            while len(buf) >= L:
                rows.append(buf[:L]); buf = buf[L:]
                if len(rows) >= n:
                    print(f"[data] real OpenWebText ({n} seqs)", flush=True)
                    return torch.tensor(rows, device=device)
    except Exception as e:
        print(f"[data] OWT failed ({e}); RANDOM tokens (plumbing only)", flush=True)
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, model.vocab_size - 1, (n, L), generator=g).to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["toy", "ckpt"], default="ckpt")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--architecture", default="small")
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--mask-frac", type=float, default=0.5)
    ap.add_argument("--max-candidates", type=int, default=16)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--ms", type=int, nargs="+", default=[8, 32, 128, 512])
    ap.add_argument("--bits", type=int, nargs="+", default=[3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="characterize_Ai_token.json")
    args = ap.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "ckpt" and not args.checkpoint:
        ap.error("--checkpoint required unless --model toy")
    if args.model == "toy":
        args.seq_len = 32
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    model = build_model(args)
    Vc, mask_id = model.vocab_size, model.mask_index
    L = min(args.seq_len, getattr(model, "seq_len", args.seq_len))
    x0 = real_text(model, args.n_seqs, L, args.device, args.seed)

    pr_all, H_all, n90_all = [], [], []
    curv, curv_null, gini_all, gain_all = [], [], [], []
    Ai = {(m, B): [] for m in args.ms for B in args.bits}
    for s in range(args.n_seqs):
        x = x0[s:s + 1]
        masked = (torch.rand(x.shape, device=args.device, generator=gen)
                  < args.mask_frac)
        masked[:, 0] = False
        z = torch.where(masked, mask_id, x)
        p = posteriors(model, z)[0]                      # (L, Vc)
        mrows = masked[0]
        pm = p[mrows].cpu().numpy()                      # (n_masked, Vc)
        pr, H, n90 = effective_support(pm)
        pr_all += pr.tolist(); H_all += H.tolist(); n90_all += n90.tolist()

        # lift the true token per masked position
        xm = x[0][mrows]
        v = args.a * F.one_hot(xm, Vc).float()
        v = (v + args.b * (-torch.log(-torch.log(torch.rand(
            v.shape, device=args.device, generator=gen).clamp_min(1e-12))))
             ).cpu().numpy()
        curv += curvature_full(v, pm, args.a, args.b).tolist()
        vsh = np.take_along_axis(v, np.argsort(
            np.random.default_rng(s).random(v.shape), axis=-1), axis=-1)
        curv_null += curvature_full(vsh, pm, args.a, args.b).tolist()
        gini_all += gini(pm).tolist()
        for (m, B) in Ai:
            Ai[(m, B)] += curvature_from_Ai(v, pm, args.a, args.b, m, B).tolist()

        # commitment_gain via MDLM leave-one-out on a candidate subset
        cand = mrows.nonzero(as_tuple=True)[0]
        if len(cand) > args.max_candidates:
            cand = cand[torch.randperm(len(cand), device=cand.device)[:args.max_candidates]]
        base_H = entropy(p.cpu().numpy())               # (L,)
        gains = np.full(int(mrows.sum()), np.nan)
        midx = {int(j): k for k, j in enumerate(mrows.nonzero(as_tuple=True)[0].tolist())}
        for j in cand.tolist():
            zc = z.clone(); zc[0, j] = int(p[j].argmax())
            pj = posteriors(model, zc)[0].cpu().numpy()
            H2 = entropy(pj)
            still = (zc[0] == mask_id).cpu().numpy()
            gains[midx[j]] = ((base_H * still).sum() - base_H[j]) \
                - ((H2 * still).sum() - H2[j])
        gain_all += gains.tolist()

    pr_all = np.array(pr_all); n90_all = np.array(n90_all)
    gv = np.array(gini_all); gain = np.array(gain_all)
    full = within_bin_spearman(-np.array(curv), gain, gv)
    null = within_bin_spearman(-np.array(curv_null), gain, gv)
    report = {
        "effective_support": {
            "participation_ratio_median": round(float(np.median(pr_all)), 1),
            "participation_ratio_p90": round(float(np.percentile(pr_all, 90)), 1),
            "entropy_median_nats": round(float(np.median(H_all)), 3),
            "n_tokens_for_90pct_median": int(np.median(n90_all)),
            "n_tokens_for_90pct_p90": int(np.percentile(n90_all, 90)),
            "vocab_size": int(Vc),
        },
        "residual": {"full_corr": round(full, 4), "null_corr": round(null, 4),
                     "grid": {}},
    }
    print(json.dumps(report["effective_support"], indent=2))
    print(f"\nresidual: full={full:+.3f} null={null:+.3f}")
    print(f"  A_i state         | corr    | % of full | beats null")
    for m in args.ms:
        for B in args.bits:
            c = within_bin_spearman(-np.array(Ai[(m, B)]), gain, gv)
            frac = 100 * (c / full) if full else 0
            beats = (c * full > 0) and abs(c) > 1.3 * abs(null)
            report["residual"]["grid"][f"m{m}_B{B}"] = {
                "corr": round(c, 4), "pct_of_full": round(frac, 1),
                "beats_null": bool(beats)}
            print(f"  top-{m:<4} {B}-bit      | {c:+.3f} | {frac:6.0f}%  | "
                  f"{'YES' if beats else 'no'}")
    json.dump(report, open(args.output, "w"), indent=2)
    print(f"\n[token-gate] wrote {args.output}")


if __name__ == "__main__":
    main()
