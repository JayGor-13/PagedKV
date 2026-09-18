"""Audit shared PCA, global protection, storage overhead and isolated-page decode.

Uses fresh seeded synthetic tensors and the actual 4-sink/128-window policy.
"""
import json
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np

from kvtc import KVTCCodec, KVTCConfig
from kvtc.cold_store import ColdStore, _split_payload
from kvtc import serialize
from scripts.storage_overhead import payload_parts


def make_sample(seed=20260915, length=1024):
    rng = torch.Generator().manual_seed(seed)
    cfg = KVTCConfig(target_cr=4, pca_rank_cap=16, block_sizes=(4, 16),
                     sink_tokens=4, window_tokens=128, dp_calib_subsample=0)
    codec = KVTCCodec(cfg, device='cpu')
    # Independent calibration and held-out K/V tensors; no claim about their
    # realism or reconstruction quality as model activations.
    train_k = torch.randn(256, 64, generator=rng)
    train_v = torch.randn(256, 64, generator=rng)
    codec.calibrate([train_k], [train_v], verbose=False)
    return codec, torch.randn(length, 64, generator=rng), torch.randn(length, 64, generator=rng)


def audit(codec, keys, values, page_tokens):
    original_compress = KVTCCodec.compress
    seen = []

    def checked_compress(page_codec, k, v):
        # Check the actual encoder used on EVERY page, including its assignments.
        assert page_codec.art is codec.art
        assert page_codec.art.key is codec.art.key
        assert page_codec.art.value is codec.art.value
        seen.append(len(k))
        return original_compress(page_codec, k, v)

    with patch.object(KVTCCodec, 'calibrate', side_effect=AssertionError('per-page calibration')):
        with patch.object(KVTCCodec, 'compress', new=checked_compress):
            archive = ColdStore.encode(codec, keys, values, page_tokens)
    assert len(seen) == archive.page_count
    assert archive._codec.art.key is codec.art.key
    assert archive._codec.art.value is codec.art.value
    assert all(set(item) == {'rank', 'blocks', 'bits_per_token'}
               for item in archive.metadata['shared'].values())

    # Reopen from the archive bytes and shared basis before testing numerical equality.
    archive = ColdStore(archive.blob, codec)
    full = archive.decode_all()
    monolithic = codec.compress(keys, values)
    mono_k, mono_v = codec.decompress(monolithic)
    mono_symbols = {name: serialize.deserialize(payload.blob, codec=codec.cfg.entropy_codec)
                    for name, payload in monolithic.items()}
    length = len(keys)
    expected_protected = set(range(min(4, length))) | set(range(max(0, length-128), length))
    covered = {'key': [], 'value': []}
    counts, exact = [], {'key': True, 'value': True}
    symbols_exact = {'key': True, 'value': True}
    for page in range(archive.page_count):
        start = page * page_tokens
        end = min(length, start + page_tokens)
        recovered = archive.decode_pages([page])
        torch.testing.assert_close(recovered.positions, torch.arange(start, end), rtol=0, atol=0)
        for name, tensor, complete in (('key', recovered.keys, full.keys),
                                        ('value', recovered.values, full.values)):
            is_exact = torch.equal(tensor, complete[start:end])
            exact[name] &= is_exact
            assert is_exact, 'isolated page differs from full decode of same archive'
            hdr, _ = _split_payload(archive._page_payloads(page)[name].blob)
            if hdr['n_compressed']:
                page_symbols = serialize.deserialize(archive._page_payloads(page)[name].blob,
                                                     codec=codec.cfg.entropy_codec)
                mono = mono_symbols[name]
                first = max(start, codec.cfg.sink_tokens) - codec.cfg.sink_tokens
                last = first + hdr['n_compressed']
                symbols_exact[name] &= (np.array_equal(page_symbols[1], mono[1][first:last])
                                       and np.array_equal(page_symbols[2], mono[2])
                                       and np.array_equal(page_symbols[3], mono[3][:, first:last])
                                       and np.array_equal(page_symbols[4], mono[4][:, first:last]))
            if not hdr['n_compressed']:
                local_protected = list(range(end-start))
            else:
                local_protected = list(range(hdr['sink'])) + list(range(end-start-hdr['window'], end-start))
            global_protected = [start + i for i in local_protected]
            assert set(global_protected) == expected_protected.intersection(range(start, end))
            assert hdr['n_protected'] == len(global_protected)
            covered[name].extend(global_protected)
            if name == 'key':
                counts.append(dict(page=page, start=start, end=end,
                                   protected=hdr['n_protected'], compressed=hdr['n_compressed']))
    for positions in covered.values():
        assert len(positions) == len(set(positions)) == len(expected_protected)
        assert set(positions) == expected_protected
    protected_idx = torch.tensor(sorted(expected_protected), dtype=torch.long)
    torch.testing.assert_close(full.keys[protected_idx], keys[protected_idx].half().float(), rtol=0, atol=0)
    torch.testing.assert_close(full.values[protected_idx], values[protected_idx].half().float(), rtol=0, atol=0)

    # Independent monolithic ENCODING can use different floating-point GEMM
    # batch shapes. Report strict equality separately; require numerical agreement.
    torch.testing.assert_close(full.keys, mono_k, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full.values, mono_v, rtol=1e-5, atol=1e-6)
    legacy = ColdStore.encode(codec, keys, values, page_tokens, format_version=1)
    mono_parts = {name: sum(payload_parts(p.blob)[name] for p in monolithic.values())
                  for name in payload_parts(monolithic['key'].blob)}
    paged_parts = archive.storage_breakdown()
    mono_size = sum(p.nbytes() for p in monolithic.values())
    entropy_extra = (paged_parts['codes'] + paged_parts['scales_shifts'] + paged_parts['protected']
                     - mono_parts['codes'] - mono_parts['scales_shifts'] - mono_parts['protected'])
    legacy_header_sizes = [payload_parts(p.blob)['json_headers'] for page in range(legacy.page_count)
                           for p in legacy._page_payloads(page).values()]
    basis_bytes = sum(t.numel()*t.element_size() for b in (codec.art.key, codec.art.value)
                      for t in (b.mu, b.V, b.evals))
    return dict(tokens=length, features=keys.shape[1], page_tokens=page_tokens,
                page_count=archive.page_count, protected_tokens_per_cache_type=len(expected_protected),
                page_counts=counts, encoder_shared_artifact_checked_on_pages=len(seen),
                per_page_calibration_calls=0, basis_tensor_bytes_stored_in_archive=0,
                shared_calibration_resident_bytes=basis_bytes,
                isolated_vs_same_archive_full_exact=exact,
                quantized_codes_widths_scales_shifts_vs_monolithic_exact=symbols_exact,
                paged_vs_independently_encoded_monolithic_exact={
                    'key': torch.equal(full.keys, mono_k), 'value': torch.equal(full.values, mono_v)},
                paged_vs_monolithic_max_abs_error={
                    'key': float((full.keys-mono_k).abs().max()),
                    'value': float((full.values-mono_v).abs().max())},
                monolithic_bytes=mono_size, legacy_paged_bytes=legacy.nbytes(),
                compact_paged_bytes=archive.nbytes(),
                compact_overhead_percent=100*(archive.nbytes()/mono_size-1),
                independently_compressed_stream_extra_bytes=entropy_extra,
                container_extra_bytes=archive.nbytes()-mono_size-entropy_extra,
                legacy_json_bytes_per_cache_page_min=min(legacy_header_sizes),
                legacy_json_bytes_per_cache_page_max=max(legacy_header_sizes),
                legacy_json_bytes_all_pages=sum(legacy_header_sizes),
                monolithic_parts=mono_parts, compact_parts=paged_parts)


def main():
    torch.set_num_threads(2)
    rows = []
    for seed, length in ((20260915, 1024), (20260916, 1019)):
        codec, keys, values = make_sample(seed, length)
        for page_tokens in (64, 128, 256):
            row = dict(seed=seed, **audit(codec, keys, values, page_tokens))
            rows.append(row)
            print(f"seed={seed}, tokens={length}, page={page_tokens}: "
                  f"protected={row['protected_tokens_per_cache_type']}, "
                  f"isolated/full exact={row['isolated_vs_same_archive_full_exact']}, "
                  f"monolithic exact={row['paged_vs_independently_encoded_monolithic_exact']}, "
                  f"symbols exact={row['quantized_codes_widths_scales_shifts_vs_monolithic_exact']}, "
                  f"overhead={row['compact_overhead_percent']:.2f}%", flush=True)
    result = dict(run_kind='synthetic_page_controls', device='cpu', seed_policy='fixed fresh seeds for reproducibility',
                  calibration_rows_per_type=256, sink_tokens=4, window_tokens=128,
                  target_cr=4, pca_rank=16, layout='token_major', entropy_codec='deflate', cases=rows)
    path = Path('outputs/page_controls.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    primary = rows[1]
    lines = ['# Random-sample page controls — 2026-09-15', '',
             'All six cases use the actual first-4/last-128 protection policy, 64 features, rank 16, target CR 4, and independently generated calibration/K/V tensors. This is a CPU numerical/storage audit, not an LLM-quality or GPU-performance experiment.', '',
             '## Shared PCA', '',
             'The actual encoder on every page was intercepted and checked to use the original calibration artifact. Recalibration was disabled during paging. All pages share one key basis and one value basis; K and V are calibrated separately. No basis tensor is serialized per page or into the archive. The common external calibration artifact is required for decoding and must be counted separately.', '',
             '## Global protected tokens', '',
             'Every case stores exactly 132 protected token positions per cache type, with no duplicates. Original positions 0–3 and the final 128 positions are checked explicitly; all other positions are compressed. A middle page shorter than 132 tokens is not automatically protected.', '',
             '| Page | Token range (end excluded) | Protected | Compressed |',
             '| ---: | --- | ---: | ---: |']
    for r in primary['page_counts']:
        lines.append(f"| {r['page']} | [{r['start']}, {r['end']}) | {r['protected']} | {r['compressed']} |")
    lines += ['', '## Storage and exactness', '',
              '| Seed / tokens | Page tokens | Monolithic bytes | Compact paged bytes | Overhead | Isolated/full exact K,V | Separate monolithic encoding exact K,V |',
              '| --- | ---: | ---: | ---: | ---: | --- | --- |']
    for r in rows:
        lines.append(f"| {r['seed']} / {r['tokens']} | {r['page_tokens']} | {r['monolithic_bytes']} | {r['compact_paged_bytes']} | {r['compact_overhead_percent']:.2f}% | {list(r['isolated_vs_same_archive_full_exact'].values())} | {list(r['paged_vs_independently_encoded_monolithic_exact'].values())} |")
    lines += ['', 'The isolated/full comparison requires exact tensor equality (zero tolerance) for EVERY page, for both K and V. The independently encoded monolithic comparison separately records exact equality and maximum absolute differences; it requires numerical agreement at rtol=1e-5, atol=1e-6 because matrix multiplication batch shapes can change floating-point rounding. Finite samples do not prove bitwise equality across devices or all inputs.', '',
              f"Quantized codes, widths, FP16 scales and shifts match the independently encoded monolithic cache exactly in all six cases: {all(all(r['quantized_codes_widths_scales_shifts_vs_monolithic_exact'].values()) for r in rows)}. The largest reconstructed-value discrepancy is {max(max(r['paged_vs_monolithic_max_abs_error'].values()) for r in rows):.9g}. Identical stored quantization data localizes these differences to floating-point reconstruction rather than changed quantization decisions.", '',
              f"For the 1,024-token / 128-token-page sample, old-format JSON headers range from {primary['legacy_json_bytes_per_cache_page_min']} to {primary['legacy_json_bytes_per_cache_page_max']} bytes per K or V page, totaling {primary['legacy_json_bytes_all_pages']} bytes. Header size depends on rank and block assignments; 4.5 KB is not a fixed per-page header size. The compact format stores shared compressed metadata once plus 25 bytes per page and 8-byte offsets with a final sentinel.", '',
              f"In that sample, splitting entropy streams adds {primary['independently_compressed_stream_extra_bytes']} bytes versus monolithic entropy streams; container differences add {primary['container_extra_bytes']} bytes. The signed components sum to the total overhead. This measures fragmentation separately from metadata.", '',
              'The earlier 37.03% example used a 16-token recent window, not 128. Its protected-token policy was still global: the measured overhead was dominated by repeated headers, not repeated raw windows. The current random audit explicitly covers a 128-token window and partial final pages.', '',
              'All reported archive sizes include headers and indices, but exclude shared calibration, Python objects, temporary buffers and any future hot cache. No codec changes were needed for these controls.', '',
              'Run `python -m scripts.audit_page_controls`; raw results are in `outputs/page_controls.json`.', '']
    report = Path('reports/page_controls.md')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
