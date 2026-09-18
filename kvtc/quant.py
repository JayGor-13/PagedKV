"""Quantizers used by the KVTC dynamic program.

Paper Appendix B.17: types = [None, int2, int4, fp8], applied to contiguous
blocks of PCA coordinates, each block carrying a shared 16-bit shift and scale.

CRITICAL IMPLEMENTATION DETAIL
------------------------------
The encoder must quantize using the *fp16-rounded* scale and shift, because the
decoder only ever sees the fp16 values. Quantizing with fp32 scales and then
storing fp16 ones makes encode and decode disagree -- a bug that shows up as a
small, hard-to-trace reconstruction error. Every function here rounds scale and
shift to fp16 BEFORE using them.
"""
import torch
from .config import QUANT_BITS

FP8_MAX = 448.0  # torch.float8_e4m3fn max representable magnitude
FP16_MIN_POSITIVE = 2.0 ** -24  # smallest positive FP16 subnormal


def used_bits(qtype: str, block_size: int, meta_bits: int = 32) -> int:
    """Bits per TOKEN to code one block of `block_size` PCA coordinates."""
    if qtype == "none":
        return 0
    return block_size * QUANT_BITS[qtype] + meta_bits


def quantize_block(x: torch.Tensor, qtype: str):
    """Quantize one block.

    Args:
        x: (n_tokens, block_size) float32
        qtype: one of 'none' | 'int2' | 'int4' | 'fp8'

    Returns:
        deq:   (n_tokens, block_size) float32, the reconstruction
        codes: (n_tokens, block_size) uint8 integer codes ('none' -> empty)
        scale: (n_tokens, 1) float16
        shift: (n_tokens, 1) float16
    """
    n = x.shape[0]
    dev = x.device
    if qtype == "none":
        return (torch.zeros_like(x),
                torch.zeros((n, 0), dtype=torch.uint8, device=dev),
                torch.zeros((n, 1), dtype=torch.float16, device=dev),
                torch.zeros((n, 1), dtype=torch.float16, device=dev))

    if qtype == "fp8":
        amax = x.abs().amax(dim=1, keepdim=True)
        scale16 = (amax / FP8_MAX).clamp_min(FP16_MIN_POSITIVE).to(torch.float16)
        s = scale16.float()
        q = (x / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        deq = q.float() * s
        codes = q.view(torch.uint8)
        shift16 = torch.zeros((n, 1), dtype=torch.float16, device=dev)
        return deq, codes, scale16, shift16

    # int2 / int4: uniform (KIVI-style) quantization with shared shift+scale
    b = QUANT_BITS[qtype]
    levels = (1 << b) - 1
    lo = x.amin(dim=1, keepdim=True)
    hi = x.amax(dim=1, keepdim=True)
    shift16 = lo.to(torch.float16)
    scale16 = ((hi - lo) / levels).clamp_min(FP16_MIN_POSITIVE).to(torch.float16)
    s, sh = scale16.float(), shift16.float()
    codes = torch.round((x - sh) / s).clamp_(0, levels).to(torch.uint8)
    deq = codes.float() * s + sh
    return deq, codes, scale16, shift16


def dequantize_block(codes: torch.Tensor, scale16: torch.Tensor,
                     shift16: torch.Tensor, qtype: str, block_size: int):
    """Exact inverse of quantize_block, from stored fp16 metadata only."""
    n = scale16.shape[0]
    dev = scale16.device
    if qtype == "none":
        return torch.zeros((n, block_size), dtype=torch.float32, device=dev)
    s = scale16.float()
    if qtype == "fp8":
        q = codes.view(torch.float8_e4m3fn)
        return q.float() * s
    return codes.float() * s + shift16.float()


def block_sq_error(x: torch.Tensor, qtype: str) -> float:
    """Squared Frobenius error from quantizing block `x` with `qtype`."""
    if qtype == "none":
        return float((x * x).sum())
    deq, _, _, _ = quantize_block(x, qtype)
    d = x - deq
    return float((d * d).sum())
