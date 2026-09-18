"""Reproducible storage ablation on synthetic caches; no model or latency claim."""
import json
from pathlib import Path
import struct

import torch

from kvtc import KVTCCodec, KVTCConfig
from kvtc.cold_store import ColdStore


def fixture(features=16, length=1024, correlated=False, seed=31):
    rng = torch.Generator().manual_seed(seed)
    cfg = KVTCConfig(target_cr=4, pca_rank_cap=min(16, features), block_sizes=(4, 16),
                     sink_tokens=4, window_tokens=16, dp_calib_subsample=0)
    codec = KVTCCodec(cfg, device='cpu')
    if correlated:
        mapping = torch.randn(6, features, generator=rng)

        def sample(n):
            return torch.randn(n, 6, generator=rng) @ mapping + .02 * torch.randn(n, features, generator=rng)
    else:
        def sample(n):
            return torch.randn(n, features, generator=rng)
    train = sample(128)
    codec.calibrate([train], [train * .7], verbose=False)
    return codec, sample(length), sample(length)


def payload_parts(blob):
    off, lengths = 5, []
    for _ in range(4):
        size, = struct.unpack_from('<Q', blob, off)
        lengths.append(size)
        off += 8 + size
    return dict(zip(('json_headers', 'codes', 'scales_shifts', 'protected'), lengths),
                framing=37)


def legacy_breakdown(archive):
    parts = dict(index=archive.index_bytes, json_headers=0, codes=0,
                 scales_shifts=0, protected=0, framing=0)
    for _, _, offset, nk, nv in archive._entries:
        for blob in (archive.blob[offset:offset+nk], archive.blob[offset+nk:offset+nk+nv]):
            for name, size in payload_parts(blob).items():
                parts[name] += size
    assert sum(parts.values()) == archive.nbytes()
    return parts


def write_report(result):
    original = result['monolithic_bytes']
    old, new = result['old_paged_bytes'], result['compact_bytes']
    lines = [
        '# Selective-decompression storage ablation', '',
        'Measured locally on CPU on 2026-09-14. These are synthetic storage and numerical-correctness results, not LLM accuracy or speed results.', '',
        '## Same workload, same 128-token pages', '',
        '| Format | Stored bytes | Overhead versus original monolithic KVTC |',
        '| --- | ---: | ---: |',
        f'| Original monolithic | {original:,} | 0% |',
        f'| Original paged format (v1) | {old:,} | {100*(old/original-1):.2f}% |',
        f'| Compact paged format (v2) | {new:,} | {100*(new/original-1):.2f}% |', '',
        f'The new format removes {old-new:,} bytes: {100*(old-new)/(old-original):.1f}% of the original paging overhead, or {100*(old-new)/old:.1f}% of the old paged archive size.', '',
        'The v2 format stores a single zlib-compressed metadata record for the archive, an array of 64-bit page offsets, and a 25-byte record per page containing protected-value dtype flags and six stream lengths. Page token ranges are derived from page size. Empty streams are omitted. Nonempty entropy streams remain byte-for-byte unchanged.', '',
        'This is a container change. PCA calibration, quantizer choices, precision, protection regions and page size are unchanged. The decoder rebuilds transient legacy headers for selected pages to reuse the original reconstruction path. That adds allocation work; latency and peak memory have not been measured. Encoding currently creates legacy payloads before compacting them, so its transient memory use is not optimized.', '',
        '## Stored-byte breakdown', '',
        '| Part | Original paged v1 | Compact paged v2 |',
        '| --- | ---: | ---: |',
    ]
    for label, a, b in [('Shared metadata and index', 'index', 'index'),
                         ('Repeated page JSON', 'json_headers', 'page_headers'),
                         ('Page framing', 'framing', 'page_framing'),
                         ('Quantized codes', 'codes', 'codes'),
                         ('Scales and shifts', 'scales_shifts', 'scales_shifts'),
                         ('Protected entries', 'protected', 'protected')]:
        lines.append(f"| {label} | {result['old_paged_parts'][a]:,} | {result['compact_parts'][b]:,} |")
    lines += ['', '## Page-size and data sensitivity', '',
              'Calibration uses 128 synthetic rows, target CR 4, at most 16 PCA coordinates, token-major serialization, four sink tokens and a 16-token recent window. These small feature dimensions are not representative of the full model-wide KVTC basis. There are two independent Gaussian seeds and one 64-feature correlated workload; no representative real-cache distribution has been evaluated.', '',
              'Requested token locations are fixed at 129 and 641 for every page size. All pages containing those two locations are decoded. Thus larger pages also recover more unrelated tokens.', '',
              '| Features / tokens / seed / data | Page tokens | v1 overhead | v2 overhead | v2 overhead vs compact one-page control | Tokens decoded |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in result['sweep']:
        kind = 'correlated' if r['correlated'] else 'Gaussian'
        lines.append(f"| {r['features']} / {r['tokens']} / {r['seed']} / {kind} | {r['page_tokens']} | {r['old_overhead_percent']:.2f}% | {r['compact_overhead_percent']:.2f}% | {r['compact_vs_one_page_percent']:.2f}% | {r['selected_tokens']} |")
    lines += ['', 'The compact one-page control applies the same metadata savings to full-cache storage. It prevents attributing all improvement to paging. Negative overhead relative to the original monolithic format can arise from cheaper metadata; it does not establish a fundamental compression advantage.', '',
              'Every sweep row checks exact v1/v2 reconstructed K/V equality and exact selected/full-page equality within v2. Unit tests additionally cover both layouts, identity and DEFLATE coding, empty/short/partial pages, all-dropped coordinates, FP16/BF16 protected entries, invalid IDs, and selective recovery despite corruption of an unselected page.', '',
              'All sizes include stored archive metadata and page indices. Shared calibration tensors, Python object overhead, transient buffers and a future hot cache are excluded. All archives use the same external calibration artifact. GPU performance, query selection and model attention remain unimplemented.', '',
              '## Next measurements', '',
              'Keep 128-token pages as the current development setting; measure 64/128/256 on actual model caches before selecting a deployment size. If entropy-stream fragmentation remains material, evaluate shared calibration-derived dictionaries with their storage cost counted, or independently decompressible subpages within larger indexed groups. Larger groups must be charged for all bytes and tokens decoded. These alternatives have not been implemented.', '',
              'Reproduce with `python -m scripts.storage_overhead`. Raw measurements are written to `outputs/storage_overhead.json`.', '']
    path = Path('reports/storage_overhead.md')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    torch.set_num_threads(2)
    codec, k, v = fixture()
    mono = codec.compress(k, v)
    archive = ColdStore.encode(codec, k, v, page_tokens=128, format_version=1)
    compact = ColdStore.encode(codec, k, v, page_tokens=128)
    result = dict(run_kind='synthetic_storage_ablation',
                  monolithic_bytes=sum(x.nbytes() for x in mono.values()),
                  monolithic_parts={name: sum(payload_parts(x.blob)[name] for x in mono.values())
                                    for name in payload_parts(mono['key'].blob)},
                  old_paged_bytes=archive.nbytes(), old_paged_parts=legacy_breakdown(archive),
                  compact_bytes=compact.nbytes(), compact_parts=compact.storage_breakdown(),
                  excluded='Shared calibration, runtime buffers, Python overhead, hot cache; no timing',
                  sweep=[])
    cases = [(16, 1024, False, 31), (16, 1024, False, 37), (64, 4096, True, 31)]
    for features, length, correlated, seed in cases:
        if (features, length, correlated, seed) != cases[0]:
            codec, k, v = fixture(features, length, correlated, seed)
        monolithic = codec.compress(k, v)
        mono_bytes = sum(p.nbytes() for p in monolithic.values())
        compact_one_page = ColdStore.encode(codec, k, v, length).nbytes()
        for page_tokens in (32, 64, 128, 256, 512):
            old = ColdStore.encode(codec, k, v, page_tokens, format_version=1)
            new = ColdStore.encode(codec, k, v, page_tokens)
            old_full, new_full = old.decode_all(), new.decode_all()
            torch.testing.assert_close(old_full.keys, new_full.keys, rtol=0, atol=0)
            torch.testing.assert_close(old_full.values, new_full.values, rtol=0, atol=0)
            # Keep requested token locations fixed when varying page size.
            ids = sorted({129 // page_tokens, 641 // page_tokens})
            selected = new.decode_pages(ids)
            torch.testing.assert_close(selected.keys, new_full.keys[selected.positions], rtol=0, atol=0)
            torch.testing.assert_close(selected.values, new_full.values[selected.positions], rtol=0, atol=0)
            row = dict(features=features, tokens=length, correlated=correlated, seed=seed,
                       page_tokens=page_tokens, monolithic_bytes=mono_bytes,
                       old_bytes=old.nbytes(), compact_bytes=new.nbytes(),
                       old_overhead_percent=100*(old.nbytes()/mono_bytes-1),
                       compact_overhead_percent=100*(new.nbytes()/mono_bytes-1),
                       compact_one_page_bytes=compact_one_page,
                       compact_vs_one_page_percent=100*(new.nbytes()/compact_one_page-1),
                       selected_tokens=len(selected.positions), selected_payload_bytes=selected.payload_bytes_read,
                       index_bytes=new.index_bytes, exact_legacy_page_reconstruction=True)
            result['sweep'].append(row)
            print(f"p={features}, n={length}, seed={seed}, page={page_tokens}: "
                  f"overhead {row['old_overhead_percent']:.2f}% -> {row['compact_overhead_percent']:.2f}%; "
                  f"decoded {row['selected_tokens']} tokens", flush=True)
    path = Path('outputs/storage_overhead.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    write_report(result)
    print(f'Report: {path.resolve()}')


if __name__ == '__main__':
    main()
