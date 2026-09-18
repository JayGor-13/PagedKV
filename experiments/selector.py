"""Reference-like page ranking from STORED quantized key coefficients.

This baseline scans/dequantizes all key streams. It is not a zero-cost selector:
it allocates N x topk FP32 coefficients and temporarily reconstructs low-rank
per-head keys. Only selected pages undergo full K/V reconstruction afterward.
"""
import numpy as np
import torch

from kvtc import serialize
from kvtc.quant import dequantize_block
from kvtc.cold_store import _PAGE


@torch.no_grad()
def scan_key_coefficients(archive, topk):
    basis = archive._codec.art.key
    rank = min(int(topk), basis.V.shape[1])
    if rank < 1:
        raise ValueError('topk must be positive')
    coefficients = torch.zeros(archive.metadata['seq_len'], rank)
    payload_bytes = 0
    for page_id, (start, end, offset, nk, nv) in enumerate(archive._entries):
        payload = archive._page_payloads(page_id, 'key')['key']
        hdr, codes, widths, scales, shifts, protected = serialize.deserialize(
            payload.blob, codec=archive.metadata['entropy_codec'])
        if archive.format_version == 1:
            payload_bytes += nk
        else:
            _, *sizes = _PAGE.unpack_from(archive.blob, offset)
            payload_bytes += _PAGE.size + sum(sizes[:3])
        n, sink, window = hdr['n_compressed'], hdr['sink'], hdr['window']
        col, bi = 0, 0
        for a, b, quant in hdr['blocks']:
            if quant == 'none':
                continue
            if n and a < rank:
                c = torch.from_numpy(np.ascontiguousarray(codes[:, col:col+b-a]))
                sc = torch.from_numpy(scales[bi].copy())[:, None]
                sh = torch.from_numpy(shifts[bi].copy())[:, None]
                deq = dequantize_block(c, sc, sh, quant, b-a)
                coefficients[start+sink:start+sink+n, a:min(b, rank)] = deq[:, :min(b, rank)-a]
            col += b-a
            bi += 1
        if len(protected):
            positions = (torch.arange(start, end) if not n else
                         torch.cat([torch.arange(start, start+sink), torch.arange(end-window, end)]))
            # Protected keys have no stored coefficients; project their decoded
            # 16-bit values, rather than consulting the original full cache.
            p = torch.from_numpy(protected).to(basis.mu.device)
            coefficients[positions] = ((p-basis.mu) @ basis.V[:, :rank]).cpu()
    return coefficients, dict(key_scan_payload_bytes=payload_bytes,
                              coefficient_buffer_bytes=coefficients.numel()*coefficients.element_size(),
                              key_pages_scanned=archive.page_count,
                              values_scanned=False, shared_index_bytes=archive.index_bytes)
