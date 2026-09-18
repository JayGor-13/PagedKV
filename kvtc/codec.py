"""Top-level KVTC codec: calibrate -> compress -> decompress.

Paper Section 3, three modes of operation:
  Calibration   once per model. Fit PCA (K and V separately), run the DP for
                each target compression ratio. ONE PCA basis is shared across
                all ratios (Appendix B.1); only the precision assignment changes.
  Compression   project -> quantize per the assignment -> entropy code.
  Decompression reverse.

Compression ratio convention (paper Section 4 'Methods'):
    "For all methods, we calculate CR only on the compressed tokens, not
     counting the sliding window tokens."
`compression_ratio()` reports BOTH that convention and the honest end-to-end
number including protected tokens and all metadata, because they differ a lot
on short contexts and only the first is comparable to the paper's tables.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch

from .config import KVTCConfig, QUANT_BITS
from .dp import Assignment, assign_precision
from .pca import fit_pca_gram, fit_pca_randomized
from .quant import quantize_block, dequantize_block
from . import serialize as ser


@dataclass
class Basis:
    """Calibration artifact for one cache type (keys or values)."""
    mu: torch.Tensor          # (p,)
    V: torch.Tensor           # (p, r)
    evals: torch.Tensor       # (r,)

    def nbytes_fp16(self) -> int:
        return (self.mu.numel() + self.V.numel()) * 2


@dataclass
class CalibrationArtifact:
    key: Basis
    value: Basis
    assignments: Dict[str, Assignment]      # 'key' / 'value'
    cfg: KVTCConfig
    p: int

    def basis_bytes(self) -> int:
        return self.key.nbytes_fp16() + self.value.nbytes_fp16()


class KVTCCodec:
    def __init__(self, cfg: Optional[KVTCConfig] = None, device="cuda"):
        self.cfg = cfg or KVTCConfig()
        self.device = device
        self.art: Optional[CalibrationArtifact] = None

    # ------------------------------------------------------------ calibration
    def calibrate(self, key_chunks, value_chunks, verbose=True):
        """key_chunks / value_chunks: iterables of (n_i, p) float tensors."""
        cfg = self.art_cfg = self.cfg
        p = key_chunks[0].shape[1]
        rank = min(cfg.pca_rank_cap, p)
        fit = fit_pca_gram if cfg.svd_method == "gram_eigh" else fit_pca_randomized

        bases, assigns = {}, {}
        for name, chunks in (("key", key_chunks), ("value", value_chunks)):
            if cfg.svd_method == "gram_eigh":
                mu, V, ev = fit(chunks, rank=rank, device=self.device)
            else:
                mu, V, ev = fit(chunks, rank, device=self.device,
                                n_iter=cfg.randomized_iters, seed=cfg.seed)
            bases[name] = Basis(mu, V, ev)

            # projected calibration data for the DP
            P = torch.cat([((c.to(self.device, torch.float32) - mu) @ V)
                           for c in chunks], dim=0)
            budget = int(round(p * 16 / cfg.target_cr))
            a = assign_precision(P, budget, cfg)
            assigns[name] = a
            if verbose:
                print(f"  [{name}] rank={V.shape[1]}  budget={budget} b/tok  -> {a.summary()}")
            del P
            if self.device == "cuda":
                torch.cuda.empty_cache()

        self.art = CalibrationArtifact(key=bases["key"], value=bases["value"],
                                       assignments=assigns, cfg=cfg, p=p)
        return self.art

    def reassign(self, target_cr: float, key_chunks, value_chunks, verbose=True):
        """Re-run the DP at a new ratio, REUSING the fitted PCA basis.

        This is the paper's protocol (Appendix B.1): "for a given model, we use
        the same PCA matrix for all compression ratios. The only change between
        compression ratios is the precision assignment."
        """
        assert self.art is not None, "calibrate() first"
        cfg = KVTCConfig(**{**self.cfg.__dict__, "target_cr": target_cr})
        p = self.art.p
        out = {}
        for name, chunks, basis in (("key", key_chunks, self.art.key),
                                    ("value", value_chunks, self.art.value)):
            P = torch.cat([((c.to(self.device, torch.float32) - basis.mu) @ basis.V)
                           for c in chunks], dim=0)
            budget = int(round(p * 16 / target_cr))
            a = assign_precision(P, budget, cfg)
            out[name] = a
            if verbose:
                print(f"  [{name}] CR={target_cr}x -> {a.summary()}")
            del P
            if self.device == "cuda":
                torch.cuda.empty_cache()
        self.art.assignments = out
        self.art.cfg = cfg
        self.cfg = cfg
        return out

    # ------------------------------------------------------------- compression
    def encode_symbols(self, X: torch.Tensor, which: str):
        """Quantize to symbols WITHOUT serializing. Public so analyses can study
        the actual symbol streams the DP produces (entropy modelling, layout
        studies) instead of re-deriving them.

        Returns dict with:
            codes   (n_comp, R) uint8   quantized symbols, coded coordinates only
            widths  (R,) uint8          bits per coded coordinate
            scales  (n_coded_blocks, n_comp) float16
            shifts  same
            protected (n_prot, p) float32
            comp_idx, prot_idx, blocks
        """
        cfg, art = self.cfg, self.art
        basis = art.key if which == "key" else art.value
        assign = art.assignments[which]
        S, p = X.shape
        s, w = cfg.sink_tokens, cfg.window_tokens

        if S <= s + w:                      # nothing to compress
            comp_idx = torch.arange(0, dtype=torch.long)
            prot_idx = torch.arange(S)
        else:
            comp_idx = torch.arange(s, S - w)
            prot_idx = torch.cat([torch.arange(0, s), torch.arange(S - w, S)])

        prot = X[prot_idx].cpu().numpy()
        n = len(comp_idx)

        coded_cols, scales, shifts, widths = [], [], [], []
        if n > 0:
            Xc = X[comp_idx].to(self.device, torch.float32)
            D = (Xc - basis.mu) @ basis.V
            for (st, en, t) in assign.blocks:
                if t == "none":
                    # 'none' costs 0 bits in the DP and reconstructs zeros, so it
                    # must NOT occupy a scale/shift slot. Writing metadata here
                    # contradicts the budget accounting and bloats the stream.
                    continue
                blk = D[:, st:en]
                _, codes, sc, sh = quantize_block(blk, t)
                scales.append(sc.squeeze(1).cpu().numpy().astype(np.float16))
                shifts.append(sh.squeeze(1).cpu().numpy().astype(np.float16))
                coded_cols.append(codes.cpu().numpy())
                widths.extend([QUANT_BITS[t]] * (en - st))
            del D, Xc

        codes = (np.concatenate(coded_cols, axis=1) if coded_cols
                 else np.zeros((n, 0), dtype=np.uint8))
        widths = np.array(widths, dtype=np.uint8)
        nb = sum(1 for _, _, t in assign.blocks if t != "none")   # CODED blocks only
        sc_arr = (np.stack(scales) if scales else np.zeros((nb, 0), np.float16))
        sh_arr = (np.stack(shifts) if shifts else np.zeros((nb, 0), np.float16))
        return dict(codes=codes, widths=widths, scales=sc_arr, shifts=sh_arr,
                    protected=prot, comp_idx=comp_idx, prot_idx=prot_idx,
                    blocks=assign.blocks, n_blocks=nb, seq_len=S, p=p,
                    rank=int(basis.V.shape[1]),
                    bits_per_token=assign.bits_per_token)

    def _encode_one(self, X: torch.Tensor, which: str) -> ser.Payload:
        """X: (S, p) float32 for one cache type."""
        cfg = self.cfg
        sym = self.encode_symbols(X, which)
        codes, widths = sym["codes"], sym["widths"]
        sc_arr, sh_arr, prot = sym["scales"], sym["shifts"], sym["protected"]
        S, p, n, nb = sym["seq_len"], sym["p"], codes.shape[0], sym["n_blocks"]
        s, w = cfg.sink_tokens, cfg.window_tokens
        assign = self.art.assignments[which]
        prot_idx = sym["prot_idx"]

        header = dict(
            which=which, p=p, seq_len=S, n_compressed=n, n_protected=len(prot_idx),
            n_blocks=nb, blocks=[[a, b, c] for a, b, c in assign.blocks],
            widths=widths.tolist(), layout=cfg.layout,
            sink=s, window=w, rank=sym["rank"],
            bits_per_token=assign.bits_per_token, target_cr=cfg.target_cr,
        )
        return ser.serialize(codes, widths, sc_arr, sh_arr, prot, header,
                             codec=cfg.entropy_codec, level=cfg.deflate_level,
                             layout=cfg.layout)

    def compress(self, K: torch.Tensor, V: torch.Tensor):
        assert self.art is not None, "calibrate() first"
        return dict(key=self._encode_one(K, "key"),
                    value=self._encode_one(V, "value"))

    # ----------------------------------------------------------- decompression
    def _decode_one(self, payload: ser.Payload, which: str) -> torch.Tensor:
        cfg, art = self.cfg, self.art
        basis = art.key if which == "key" else art.value
        hdr, codes, widths, scales, shifts, prot = ser.deserialize(
            payload.blob, codec=cfg.entropy_codec)

        S, p, n = hdr["seq_len"], hdr["p"], hdr["n_compressed"]
        out = torch.zeros(S, p, dtype=torch.float32)

        s, w = hdr["sink"], hdr["window"]
        if n == 0:
            out[:] = torch.from_numpy(prot)
            return out

        r = hdr["rank"]
        D = torch.zeros(n, r, dtype=torch.float32, device=self.device)
        col = 0
        bi = 0                      # indexes CODED blocks only (see _encode_one)
        for (st, en, t) in hdr["blocks"]:
            if t == "none":
                continue            # already zeros
            bs = en - st
            sc = torch.from_numpy(scales[bi].astype(np.float32)).to(self.device)[:, None].half()
            sh = torch.from_numpy(shifts[bi].astype(np.float32)).to(self.device)[:, None].half()
            c = torch.from_numpy(np.ascontiguousarray(codes[:, col:col + bs])).to(self.device)
            D[:, st:en] = dequantize_block(c, sc, sh, t, bs)
            col += bs
            bi += 1

        Xc = (D @ basis.V.T + basis.mu).cpu()
        comp_idx = torch.arange(s, S - w)
        prot_idx = torch.cat([torch.arange(0, s), torch.arange(S - w, S)])
        out[comp_idx] = Xc
        out[prot_idx] = torch.from_numpy(prot)
        return out

    def decompress(self, payloads) -> tuple:
        return (self._decode_one(payloads["key"], "key"),
                self._decode_one(payloads["value"], "value"))

    # ---------------------------------------------------------------- metrics
    def compression_ratio(self, payloads, S: int, p: int) -> dict:
        """Both the paper's convention and the honest end-to-end number."""
        cfg = self.cfg
        s, w = cfg.sink_tokens, cfg.window_tokens
        n_comp = max(0, S - s - w)
        total_bytes = sum(pl.nbytes() for pl in payloads.values())

        # bytes attributable to compressed tokens only (exclude protected raw)
        comp_bytes = sum(pl.stored["codes"] + pl.stored["block_meta"]
                         for pl in payloads.values())

        orig_comp = n_comp * p * 2 * 2      # bf16, K and V
        orig_all = S * p * 2 * 2
        return dict(
            cr_paper_convention=(orig_comp / comp_bytes) if comp_bytes else float("inf"),
            cr_end_to_end=(orig_all / total_bytes) if total_bytes else float("inf"),
            compressed_tokens=n_comp, protected_tokens=S - n_comp,
            total_bytes=total_bytes, payload_bytes=comp_bytes,
            original_bytes=orig_all,
        )
