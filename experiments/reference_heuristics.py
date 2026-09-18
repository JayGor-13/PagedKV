"""Reviewed heuristics adapted from the reference experiment, not official H2O/SnapKV implementations."""
import math
import torch
import torch.nn.functional as F
from kvtc.rope import apply_rope, get_cos_sin

@torch.no_grad()
def eviction_scores(model, layers, Qs, S, device, stride=16, win=64, qchunk=256):
    cfg = model.config
    HQ, HKV = cfg.num_attention_heads, cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", cfg.hidden_size // HQ)
    rep = HQ // HKV
    cos, sin = get_cos_sin(model, S, device, torch.float32)
    ar = torch.arange(S, device=device)
    h2o = torch.zeros(S, device=device)
    snap = torch.zeros(S, device=device)
    inv = 1.0 / math.sqrt(D)

    def mass(qi, kk, Q):
        q = Q[qi.cpu()].to(device, torch.float32).view(-1, HQ, D).permute(1, 0, 2)[None]
        q = apply_rope(q, cos[:, :, qi], sin[:, :, qi])[0].to(torch.bfloat16)
        lg = torch.matmul(q, kk.transpose(1, 2)).float() * inv            # (HQ, n, S)
        lg.masked_fill_((ar[None, :] > qi[:, None])[None], float("-inf"))
        return torch.softmax(lg, -1).sum((0, 1))

    qall = torch.arange(0, S, stride, device=device)
    qwin = torch.arange(S - win, S, device=device)
    for li, (k, _) in enumerate(layers):
        kk = k[0].to(device=device, dtype=torch.bfloat16).repeat_interleave(rep, 0)  # (HQ,S,D)
        for c0 in range(0, len(qall), qchunk):
            h2o += mass(qall[c0:c0 + qchunk], kk, Qs[li])
        snap += mass(qwin, kk, Qs[li])
        del kk
    snap = F.avg_pool1d(snap[None, None], 7, stride=1, padding=3)[0, 0]
    return h2o.cpu(), snap.cpu()

@torch.no_grad()
def eviction_mask(score, S, keep_n, sink=4):
    """Keep sinks + recent window (half budget) + top-scored rest. 1 = keep."""
    keep_n = max(int(keep_n), sink + 1)
    recent = max(1, (keep_n - sink) // 2)
    m = torch.zeros(S, dtype=torch.long)
    m[:sink] = 1
    m[S - recent:] = 1
    remaining = keep_n - int(m.sum())
    if remaining > 0:
        sc = score.clone()
        sc[m == 1] = -float("inf")
        top = torch.topk(sc, min(remaining, int((m == 0).sum()))).indices
        m[top] = 1
    return m

@torch.no_grad()
def page_mass_scores(model, codec, Kd, S, queries, page, topk, hot_mask, device, band=None):
    """Attention mass per page from the top-`topk` key coefficients.

    For each (layer, kv-head, query-head): k_hat = Kd[:, :topk] @ V[slice,:topk]^T + mu;
    softmax over ALL S tokens (correct normalization); mass on NON-hot tokens summed per
    page; summed over question tokens, heads and layers. hot_mask=None -> no exclusion.
    Pre-RoPE throughout: content match, position-blind -- what the cold tier stores.
    """
    cfg = model.config
    L, HQ, HKV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    rep, hd = HQ // HKV, HKV * D
    V, mu = codec.art.key.V.to(device), codec.art.key.mu.to(device)
    Dk = Kd[:, :topk].to(device)
    n_pages = math.ceil(S / page)
    pid = torch.arange(S, device=device) // page
    keep_w = torch.ones(S, device=device)
    if hot_mask is not None:
        keep_w[hot_mask.to(device).bool()] = 0.0
    tot = torch.zeros(n_pages, device=device)
    inv = 1.0 / math.sqrt(D)
    lo, hi = (0, L) if band is None else band
    for li in range(lo, hi):
        qp = queries[li].to(device)                                      # (HQ,nq,D)
        for hkv in range(HKV):
            sl = slice(li * hd + hkv * D, li * hd + (hkv + 1) * D)
            k_hat = Dk @ V[sl, :topk].T + mu[sl]                         # (S,D)
            q = qp[hkv * rep:(hkv + 1) * rep].reshape(-1, D)             # (rep*nq, D)
            a = torch.softmax((q @ k_hat.T) * inv, dim=-1).sum(0)        # (S,) mass summed over queries
            tot.index_add_(0, pid, a * keep_w)
    return tot.cpu()

@torch.no_grad()
def pages_to_mask(pages, S, page, halo):
    m = torch.zeros(S, dtype=torch.long)
    n_pages = math.ceil(S / page)
    for pg in pages:
        for h in range(pg - halo, pg + halo + 1):
            if 0 <= h < n_pages:
                m[h * page:(h + 1) * page] = 1
    return m
