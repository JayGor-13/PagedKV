"""Local grouped low-bit quality baselines, NOT official KIVI or packed kernels.

Quantized-size estimates include group tails, BF16 metadata and native residuals.
Actual inference tensors remain in model precision and are reported separately.
"""
import math
import torch


def quantized_storage_bytes(shapes, kbits, vbits, group=32, residual=128, element_size=2):
    if kbits not in (2, 4, 8) or vbits not in (2, 4, 8) or group < 1 or residual < 0:
        raise ValueError('invalid baseline quantization parameters')
    total = 0
    for batch, heads, n, d in shapes:
        r = min(n, residual)
        prefix = n-r
        # Codes have no padding; each K/V stream is byte-aligned separately.
        total += math.ceil(batch*heads*prefix*d*kbits/8)
        total += math.ceil(batch*heads*prefix*d*vbits/8)
        total += batch*heads*math.ceil(prefix/group)*d*4
        total += batch*heads*prefix*math.ceil(d/group)*4
        total += batch*heads*r*d*2*element_size
    return total


def _roundtrip(x, bits, axis):
    lo = x.amin(axis, keepdim=True).to(torch.bfloat16).float()
    hi = x.amax(axis, keepdim=True)
    scale = ((hi-lo)/(2**bits-1)).to(torch.bfloat16).float().clamp_min(1e-6)
    return ((x-lo)/scale).round().clamp(0, 2**bits-1)*scale+lo


@torch.no_grad()
def grouped_roundtrip(layers, kbits=4, vbits=2, group=32, residual=128):
    """K groups span tokens; V groups span channels. Tail groups are unpadded."""
    shapes = [tuple(k.shape) for k, _ in layers]
    size = quantized_storage_bytes(shapes, kbits, vbits, group, residual,
                                   layers[0][0].element_size())
    out = []
    for k, v in layers:
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():
            raise ValueError('nonfinite source cache')
        n = max(0, k.shape[2]-residual)
        kq, vq = k.clone(), v.clone()
        # Vectorize full groups; handle the final partial group without fake tokens.
        complete = n//group*group
        if complete:
            b,h,_,d = k.shape
            x = k[:, :, :complete].float().reshape(b,h,-1,group,d)
            kq[:, :, :complete] = _roundtrip(x, kbits, 3).reshape(b,h,complete,d).to(k.dtype)
        if complete < n:
            kq[:, :, complete:n] = _roundtrip(k[:, :, complete:n].float(), kbits, 2).to(k.dtype)
        if n:
            b,h,_,d = v.shape
            complete_d = d//group*group
            if complete_d:
                x = v[:, :, :n, :complete_d].float().reshape(b,h,n,-1,group)
                vq[:, :, :n, :complete_d] = _roundtrip(x, vbits, -1).reshape(b,h,n,complete_d).to(v.dtype)
            if complete_d < d:
                vq[:, :, :n, complete_d:] = _roundtrip(v[:, :, :n, complete_d:].float(), vbits, -1).to(v.dtype)
        if not torch.isfinite(kq).all() or not torch.isfinite(vq).all():
            raise ValueError('nonfinite baseline reconstruction')
        out.append((kq, vq))
    return out, dict(estimated_packed_bytes=size, packed_storage_implemented=False,
                     baseline_kind='local_grouped_quantization_quality_control',
                     kbits=kbits, vbits=vbits, group=group, residual=residual)
