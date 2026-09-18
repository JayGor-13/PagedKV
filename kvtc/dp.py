"""Dynamic-programming precision assignment.

Based on the pseudocode in KVTC Appendix B.17, with two optimizations:

  1. The quantization simulation is hoisted out of the budget loop. The paper's
     pseudocode recomputes `simulate_quantization` inside `for budget in ...`,
     which is O(budget) redundant work. We precompute an error table.

  2. The budget axis is quantized by the GCD of all achievable `used_bits`
     values. With meta_bits=32 and types {int2,int4,fp8} over block sizes
     {16,64,256,1024}, every cost is a multiple of 32, so this is exact --
     not an approximation. The GCD is computed at runtime, not assumed.

`dp_stride` restricts block END positions (see KVTCConfig.dp_stride). Setting
it to 1 gives the unrestricted DP. Larger strides restrict the search and
need not return the same optimum.

The DP returns blocks that TILE [0, r) completely -- gaps are filled with the
'none' type, which is exactly how the paper's pseudocode expresses "drop this
principal component" (cost 0 bits, reconstruction 0).
"""
from dataclasses import dataclass
from math import gcd
from typing import List, Tuple
import torch

from .config import KVTCConfig, QUANT_BITS
from .quant import quantize_block, used_bits, block_sq_error


@dataclass
class Assignment:
    """A precision assignment: contiguous blocks tiling [0, rank)."""
    blocks: List[Tuple[int, int, str]]      # (start, end, qtype)
    rank: int
    bits_per_token: int                     # payload + per-block metadata
    sq_error: float                         # projected-space SSE; estimated if subsampled

    def summary(self) -> str:
        from collections import Counter
        c = Counter(t for _, _, t in self.blocks)
        cov = Counter()
        for s, e, t in self.blocks:
            cov[t] += e - s
        parts = [f"{t}:{cov[t]}c/{c[t]}b" for t in ("fp8", "int4", "int2", "none") if c[t]]
        return f"{self.bits_per_token} b/tok  rank={self.rank}  " + "  ".join(parts)

    def kept_rank(self) -> int:
        """Highest coordinate index actually coded (paper trims V to this)."""
        k = 0
        for s, e, t in self.blocks:
            if t != "none":
                k = max(k, e)
        return k


def _error_table(P, grid_pos, block_units, types, stride, meta_bits):
    """err[g][bi][ti] = error_change for coding block ending at grid point g.

    error_change = quant_error - zero_bit_error, i.e. the change relative to a
    baseline where every coordinate is dropped. Matches Appendix B.17.
    """
    G = len(grid_pos) - 1
    err = torch.full((G + 1, len(block_units), len(types)),
                     float("inf"), dtype=torch.float64)
    for g in range(1, G + 1):
        end = grid_pos[g]
        for bi, bu in enumerate(block_units):
            g0 = g - bu
            if g0 < 0:
                continue
            start = grid_pos[g0]
            blk = P[:, start:end]
            if blk.shape[1] == 0:
                continue
            zero_err = float((blk * blk).sum())
            for ti, t in enumerate(types):
                if t == "none":
                    err[g, bi, ti] = 0.0          # quant_err == zero_err
                else:
                    deq, _, _, _ = quantize_block(blk, t)
                    d = blk - deq
                    err[g, bi, ti] = float((d * d).sum()) - zero_err
    return err


def assign_precision(P: torch.Tensor, budget_bits: int, cfg: KVTCConfig) -> Assignment:
    """Solve the precision assignment.

    Args:
        P: (n_calib, rank) projected, centred calibration data = (C - mu) @ V.
           Columns MUST be ordered by descending singular value.
        budget_bits: bit budget PER TOKEN across all `rank` coordinates.
    """
    assert P.dim() == 2
    n, r = P.shape
    stride = max(1, int(cfg.dp_stride))
    if n == 0 or r == 0 or budget_bits < 0:
        raise ValueError("P must be nonempty and budget_bits must be nonnegative")
    if r % stride:
        raise ValueError("rank must be divisible by dp_stride; use dp_stride=1")

    # subsample rows for error estimation (the DP needs statistics, not all data)
    if cfg.dp_calib_subsample and n > cfg.dp_calib_subsample:
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        idx = torch.randperm(n, generator=g)[:cfg.dp_calib_subsample].to(P.device)
        Psub = P[idx].contiguous()
        scale_to_full = n / cfg.dp_calib_subsample
    else:
        Psub, scale_to_full = P, 1.0

    # ---- grid over block end positions -------------------------------------
    grid_pos = list(range(0, r, stride))
    if grid_pos[-1] != r:
        grid_pos.append(r)
    G = len(grid_pos) - 1

    # block sizes expressible on the grid, in grid units
    block_units, block_feat = [], []
    for bs in cfg.block_sizes:
        if stride == 1:
            block_units.append(bs)
            block_feat.append(bs)
        elif bs % stride == 0:
            block_units.append(bs // stride)
            block_feat.append(bs)
    if not block_units:
        raise ValueError(f"no block size divisible by dp_stride={stride}")
    # 'none' must be able to skip one grid unit
    added_skip = 1 not in block_units
    if added_skip:
        block_units.insert(0, 1)
        block_feat.insert(0, stride)

    types = list(cfg.quant_types)

    # ---- cost table, and exact budget quantum ------------------------------
    cost = torch.zeros((len(block_units), len(types)), dtype=torch.long)
    for bi, bf in enumerate(block_feat):
        for ti, t in enumerate(types):
            cost[bi, ti] = used_bits(t, bf, cfg.meta_bits_per_block)
    nz = [int(c) for c in cost.flatten() if c > 0]
    quantum = 0
    for c in nz:
        quantum = gcd(quantum, c)
    quantum = max(1, quantum)
    Q = int(budget_bits) // quantum
    cost_q = (cost // quantum)

    err = _error_table(Psub, grid_pos, block_units, types, stride,
                       cfg.meta_bits_per_block)
    if added_skip:
        # This extra transition permits dropping coordinates, not introducing
        # quantized block sizes absent from the caller's configuration.
        for ti, t in enumerate(types):
            if t != "none":
                err[:, 0, ti] = float("inf")

    # ---- DP ----------------------------------------------------------------
    INF = float("inf")
    best = torch.full((G + 1, Q + 1), INF, dtype=torch.float64)
    best[0, :] = 0.0
    bp_b = torch.full((G + 1, Q + 1), -1, dtype=torch.int16)
    bp_t = torch.full((G + 1, Q + 1), -1, dtype=torch.int16)

    for g in range(1, G + 1):
        cand = torch.full((Q + 1,), INF, dtype=torch.float64)
        cb = torch.full((Q + 1,), -1, dtype=torch.int16)
        ct = torch.full((Q + 1,), -1, dtype=torch.int16)
        for bi, bu in enumerate(block_units):
            g0 = g - bu
            if g0 < 0:
                continue
            prev = best[g0]
            for ti, t in enumerate(types):
                e = float(err[g, bi, ti])
                if e == INF:
                    continue
                c = int(cost_q[bi, ti])
                if c > Q:
                    continue
                # candidate[q] = e + prev[q - c]   for q >= c
                v = prev[: Q + 1 - c] + e
                seg = cand[c:]
                upd = v < seg
                if bool(upd.any()):
                    seg[upd] = v[upd]
                    cb[c:][upd] = bi
                    ct[c:][upd] = ti
        # monotonicity along the budget axis (paper: carry forward budget-1)
        run = cand.clone()
        rb, rt = cb.clone(), ct.clone()
        for q in range(1, Q + 1):
            if run[q - 1] < run[q]:
                run[q] = run[q - 1]
                rb[q], rt[q] = rb[q - 1], rt[q - 1]
        best[g], bp_b[g], bp_t[g] = run, rb, rt

    if not torch.isfinite(best[G, Q]):
        raise RuntimeError(
            f"DP infeasible: budget {budget_bits} bits/token too small for "
            f"rank {r}. Minimum is 0 (all 'none'); check quant_types includes 'none'.")

    # ---- traceback ---------------------------------------------------------
    blocks, g, q = [], G, Q
    total_bits = 0
    guard = 0
    while g > 0:
        guard += 1
        if guard > G + 5:
            raise RuntimeError("DP traceback did not terminate")
        bi, ti = int(bp_b[g, q]), int(bp_t[g, q])
        if bi < 0:
            # budget carried forward with no block chosen at this q; step down
            q -= 1
            if q < 0:
                raise RuntimeError("DP traceback fell off the budget axis")
            continue
        bu = block_units[bi]
        t = types[ti]
        start, end = grid_pos[g - bu], grid_pos[g]
        blocks.append((start, end, t))
        total_bits += used_bits(t, end - start, cfg.meta_bits_per_block)
        q -= int(cost_q[bi, ti])
        g -= bu
    blocks.reverse()
    blocks = merge_none_runs(blocks)

    # The DP objective subtracts the all-dropped baseline. Recompute the
    # selected reconstruction error directly to avoid subtractive cancellation.
    sq_err = sum(block_sq_error(Psub[:, s:e], t) for s, e, t in blocks) * scale_to_full
    return Assignment(blocks=blocks, rank=r, bits_per_token=total_bits, sq_error=sq_err)


def merge_none_runs(blocks):
    """Collapse runs of adjacent 'none' blocks into one.

    The DP reaches a dropped tail by chaining stride-sized 'none' steps, which
    can produce hundreds of zero-cost blocks. They are free in bits but each one
    would otherwise occupy a row in the block table and (before this fix) a
    scale/shift slot in the serialized metadata. Merging is exactly equivalent:
    'none' reconstructs zeros regardless of block size.
    """
    out = []
    for (s, e, t) in blocks:
        if out and t == "none" and out[-1][2] == "none" and out[-1][1] == s:
            out[-1] = (out[-1][0], e, "none")
        else:
            out.append((s, e, t))
    return out


# ---------------------------------------------------------------- reference
def assign_precision_bruteforce(P, budget_bits, cfg) -> Assignment:
    """Exhaustive reference for tiny instances. Used only in tests."""
    n, r = P.shape
    types = list(cfg.quant_types)
    from functools import lru_cache

    errs = {}
    for end in range(1, r + 1):
        for bs in cfg.block_sizes:
            st = end - bs
            if st < 0:
                continue
            blk = P[:, st:end]
            z = float((blk * blk).sum())
            for t in types:
                if t == "none":
                    errs[(end, bs, t)] = 0.0
                else:
                    deq, _, _, _ = quantize_block(blk, t)
                    d = blk - deq
                    errs[(end, bs, t)] = float((d * d).sum()) - z

    @lru_cache(maxsize=None)
    def rec(i, b):
        if i == 0:
            return 0.0, ()
        best, arg = float("inf"), ()
        for bs in cfg.block_sizes:
            if bs > i:
                continue
            for t in types:
                c = used_bits(t, bs, cfg.meta_bits_per_block)
                if c > b:
                    continue
                sub, path = rec(i - bs, b - c)
                v = sub + errs[(i, bs, t)]
                if v < best:
                    best, arg = v, path + ((i - bs, i, t),)
        return best, arg

    e, path = rec(r, int(budget_bits))
    tb = sum(used_bits(t, en - st, cfg.meta_bits_per_block) for st, en, t in path)
    sq_err = sum(block_sq_error(P[:, st:en], t) for st, en, t in path)
    if not path and r:
        sq_err = float("inf")
    return Assignment(blocks=list(path), rank=r, bits_per_token=tb, sq_error=sq_err)
