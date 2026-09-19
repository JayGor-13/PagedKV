"""CPU demonstration: decode specified pages while leaving all others compressed."""
import json
from pathlib import Path

import torch

from kvtc import KVTCCodec, KVTCConfig
from kvtc.cold_store import ColdStore


def main():
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(31)
    cfg = KVTCConfig(target_cr=4, pca_rank_cap=16, block_sizes=(4, 16),
                     sink_tokens=4, window_tokens=16, dp_stride=1, dp_calib_subsample=0)
    codec = KVTCCodec(cfg, device='cpu')
    train = torch.randn(128, 16, generator=rng)
    codec.calibrate([train], [train * .7], verbose=False)
    k = torch.randn(1024, 16, generator=rng)
    v = torch.randn(1024, 16, generator=rng)
    archive = ColdStore.encode(codec, k, v, page_tokens=128)
    selected = archive.decode_pages([1, 5])
    # Full decoding below is a correctness control, not part of selection.
    full = archive.decode_all()
    torch.testing.assert_close(selected.keys, full.keys[selected.positions], rtol=0, atol=0)
    torch.testing.assert_close(selected.values, full.values[selected.positions], rtol=0, atol=0)
    original_bytes = sum(p.nbytes() for p in codec.compress(k, v).values())
    result = dict(run_kind='synthetic_selective_decode_correctness', device='cpu',
                  archive_format_version=archive.format_version,
                  page_selection='explicit page IDs; no relevance selector',
                  total_pages=archive.page_count, decoded_pages=len(selected.page_ids),
                  total_tokens=len(k), decoded_tokens=len(selected.positions),
                  selected_payload_bytes=selected.payload_bytes_read,
                  full_payload_bytes=full.payload_bytes_read,
                  index_bytes=archive.index_bytes, archive_bytes=archive.nbytes(),
                  monolithic_archive_bytes=original_bytes,
                  paging_overhead_bytes=archive.nbytes() - original_bytes,
                  shared_calibration_included=False,
                  selected_matches_same_archive_full_decode=True,
                  model_attention_integrated=False, latency_measured=False)
    path = Path('outputs/selective_decode.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'Report: {path.resolve()}')


if __name__ == '__main__':
    main()
