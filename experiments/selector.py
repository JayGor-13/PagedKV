"""Page ranking from independently readable quantized key heads.

Compact-v3 archives store leading key coefficients separately. Scoring reads
that head plus its quantization metadata and protected keys; key tails and all
value streams remain unread until selected-page reconstruction.
"""
import numpy as np
import torch

from kvtc.quant import dequantize_block


@torch.no_grad()
def scan_key_coefficients(archive, topk):
    basis = archive._codec.art.key
    rank = min(int(topk), basis.V.shape[1])
    if rank < 1:
        raise ValueError('topk must be positive')
    coefficients = torch.zeros(archive.metadata['seq_len'], rank)
    payload_bytes = 0
    tail_bytes_skipped = 0
    for page_id, (start, end, offset, nk, nv) in enumerate(archive._entries):
        hdr, codes, widths, scales, shifts, protected, read, skipped = archive.key_head_symbols(page_id, rank)
        payload_bytes += read
        tail_bytes_skipped += skipped
        n, sink, window = hdr['n_compressed'], hdr['sink'], hdr['window']
        col, bi = 0, 0
        for a, b, quant in hdr['blocks']:
            if quant == 'none':
                continue
            take = max(0, min(b, rank) - a)
            if n and take:
                c = torch.from_numpy(np.ascontiguousarray(codes[:, col:col+take]))
                sc = torch.from_numpy(scales[bi].copy())[:, None]
                sh = torch.from_numpy(shifts[bi].copy())[:, None]
                deq = dequantize_block(c, sc, sh, quant, take)
                coefficients[start+sink:start+sink+n, a:a+take] = deq
            col += b-a
            bi += 1
        if len(protected):
            positions = (torch.arange(start, end) if not n else
                         torch.cat([torch.arange(start, start+sink), torch.arange(end-window, end)]))
            # Protected keys have no stored coefficients; project their decoded
            # 16-bit values, rather than consulting the original full cache.
            p = torch.from_numpy(protected).to(basis.mu.device)
            coefficients[positions] = ((p-basis.mu) @ basis.V[:, :rank]).cpu()
    available = payload_bytes + tail_bytes_skipped
    return coefficients, dict(key_scan_payload_bytes=payload_bytes,
                              key_tail_payload_bytes_skipped=tail_bytes_skipped,
                              key_payload_bytes_available=available,
                              key_scan_payload_fraction=payload_bytes / available if available else 0.,
                              key_head_rank=min(rank, archive.metadata.get('key_head_rank', rank)),
                              coefficient_buffer_bytes=coefficients.numel()*coefficients.element_size(),
                              key_pages_scanned=archive.page_count,
                              values_scanned=False, shared_index_bytes=archive.index_bytes)
