from dataclasses import replace

import pytest
import torch

from kvtc import KVTCCodec, KVTCConfig
from kvtc.cold_store import ColdStore
from kvtc import serialize


@pytest.fixture(scope='module')
def calibrated():
    cfg = KVTCConfig(target_cr=2, pca_rank_cap=8, block_sizes=(2, 4, 8),
                     sink_tokens=2, window_tokens=3, dp_stride=1, dp_calib_subsample=0)
    codec = KVTCCodec(cfg, device='cpu')
    train = torch.randn(48, 8, generator=torch.Generator().manual_seed(29))
    codec.calibrate([train], [train * .7], verbose=False)
    return codec


@pytest.mark.parametrize('length', [0, 3, 7, 19])
@pytest.mark.parametrize('entropy', ['identity', 'deflate'])
@pytest.mark.parametrize('version', [1, 2, 3])
@pytest.mark.parametrize('layout', ['token_major', 'component_major'])
def test_pages_match_original_codec_and_preserve_global_protection(calibrated, length, entropy, version, layout):
    codec = KVTCCodec(replace(calibrated.cfg, entropy_codec=entropy, layout=layout), device='cpu')
    codec.art = calibrated.art
    x = torch.randn(length, 8, generator=torch.Generator().manual_seed(13))
    original = codec.decompress(codec.compress(x, x * .3))
    archive = ColdStore.encode(codec, x, x * .3, page_tokens=4, format_version=version)
    # Reopen only from bytes plus the same shared artifact: no retained K/V.
    reopened = ColdStore(archive.blob, codec)
    full = reopened.decode_all()
    torch.testing.assert_close(full.positions, torch.arange(length))
    torch.testing.assert_close(full.keys, original[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full.values, original[1], rtol=1e-5, atol=1e-6)
    assert full.payload_bytes_read + archive.index_bytes == archive.nbytes()
    assert sum(archive.storage_breakdown().values()) == archive.nbytes()
    # Global protection must not be reapplied to every page.
    protected = torch.arange(length) if length <= 5 else torch.tensor([0, 1, length-3, length-2, length-1])
    torch.testing.assert_close(full.keys[protected], x[protected].half().float(), rtol=0, atol=0)


@pytest.mark.parametrize('version', [1, 2, 3])
def test_only_selected_pages_are_deserialized(calibrated, monkeypatch, version):
    x = torch.randn(19, 8, generator=torch.Generator().manual_seed(14))
    archive = ColdStore.encode(calibrated, x, x * .3, page_tokens=4, format_version=version)
    full = archive.decode_all()
    calls = []
    deserialize = serialize.deserialize

    def spy(blob, **kwargs):
        calls.append(blob)
        return deserialize(blob, **kwargs)

    monkeypatch.setattr(serialize, 'deserialize', spy)
    selected = archive.decode_pages([3, 1, 3])
    assert selected.page_ids == (1, 3)
    assert len(calls) == 4  # Exactly K + V for each of the two selected pages.
    assert selected.payload_bytes_read == sum(archive._entries[i][3] + archive._entries[i][4]
                                               for i in selected.page_ids)
    if version == 1:
        assert selected.payload_bytes_read == sum(map(len, calls))
    assert selected.payload_bytes_read < full.payload_bytes_read
    torch.testing.assert_close(selected.positions, torch.tensor([4, 5, 6, 7, 12, 13, 14, 15]))
    torch.testing.assert_close(selected.keys, full.keys[selected.positions], rtol=0, atol=0)
    torch.testing.assert_close(selected.values, full.values[selected.positions], rtol=0, atol=0)
    calls.clear()
    empty = archive.decode_pages([])
    assert not calls and empty.payload_bytes_read == 0 and empty.keys.shape == (0, 8)


def test_invalid_page_ids_fail_before_any_decode(calibrated, monkeypatch):
    x = torch.zeros(9, 8)
    archive = ColdStore.encode(calibrated, x, x, page_tokens=4)

    def unexpected(*args, **kwargs):
        raise AssertionError('decoder should not be called')

    monkeypatch.setattr(serialize, 'deserialize', unexpected)
    for ids in ([0, 3], [-1]):
        with pytest.raises(ValueError, match='page ID'):
            archive.decode_pages(ids)


@pytest.mark.parametrize('all_dropped', [False, True])
def test_compact_format_preserves_legacy_streams_and_reconstruction(calibrated, all_dropped):
    from kvtc.dp import Assignment
    from kvtc.cold_store import _split_payload

    codec = KVTCCodec(calibrated.cfg, device='cpu')
    codec.art = replace(calibrated.art)
    if all_dropped:
        codec.art.assignments = {name: Assignment([(0, 8, 'none')], 8, 0, 0.)
                                 for name in ('key', 'value')}
    x = torch.randn(19, 8, generator=torch.Generator().manual_seed(15))
    # Exercise per-page BF16 protected-storage fallback as well as FP16.
    x[0, 0] = 70000.
    archives = [ColdStore.encode(codec, x, x * .3, page_tokens=4, format_version=v)
                for v in (1, 2)]
    old, new = [a.decode_all() for a in archives]
    torch.testing.assert_close(old.keys, new.keys, rtol=0, atol=0)
    torch.testing.assert_close(old.values, new.values, rtol=0, atol=0)
    assert archives[1].nbytes() < archives[0].nbytes()
    for page in range(archives[0].page_count):
        for name in ('key', 'value'):
            before = _split_payload(archives[0]._page_payloads(page)[name].blob)
            after = _split_payload(archives[1]._page_payloads(page)[name].blob)
            assert before == after  # Same headers and entropy streams, including empty ones.


def test_compact_subset_does_not_decode_corrupt_unselected_page(calibrated):
    from kvtc.cold_store import _PAGE_SPLIT
    x = torch.randn(19, 8, generator=torch.Generator().manual_seed(19))
    archive = ColdStore.encode(calibrated, x, x, page_tokens=4)
    corrupted = bytearray(archive.blob)
    offset = archive._entries[2][2]
    corrupted[offset + _PAGE_SPLIT.size] ^= 255  # Damage an unselected entropy stream.
    reopened = ColdStore(bytes(corrupted), calibrated)
    actual = reopened.decode_pages([1])
    torch.testing.assert_close(actual.keys, archive.decode_pages([1]).keys, rtol=0, atol=0)
    import zlib
    with pytest.raises(zlib.error):
        reopened.decode_pages([2])


@pytest.mark.parametrize('layout', ['token_major', 'component_major'])
def test_split_key_head_scans_without_reading_tail_and_full_decode_is_exact(calibrated, layout):
    from experiments.selector import scan_key_coefficients
    from kvtc.cold_store import _PAGE_SPLIT

    codec = KVTCCodec(replace(calibrated.cfg, entropy_codec='identity', layout=layout), device='cpu')
    codec.art = calibrated.art
    x = torch.randn(19, 8, generator=torch.Generator().manual_seed(23))
    v2 = ColdStore.encode(codec, x, x * .3, page_tokens=8, format_version=2)
    split = ColdStore.encode(codec, x, x * .3, page_tokens=8, format_version=3,
                             key_head_rank=4)
    expected, actual = v2.decode_all(), split.decode_all()
    torch.testing.assert_close(actual.keys, expected.keys, rtol=0, atol=0)
    torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)

    coefficients, stats = scan_key_coefficients(split, 4)
    assert coefficients.shape == (len(x), 4)
    assert stats['key_tail_payload_bytes_skipped'] > 0
    assert stats['key_payload_bytes_available'] == (stats['key_scan_payload_bytes']
                                                     + stats['key_tail_payload_bytes_skipped'])
    assert 0 < stats['key_scan_payload_fraction'] < 1
    full_key_bytes = 0
    for _, _, offset, _, _ in split._entries:
        _, *sizes = _PAGE_SPLIT.unpack_from(split.blob, offset)
        full_key_bytes += _PAGE_SPLIT.size + sum(sizes[:4])
    assert stats['key_scan_payload_bytes'] < full_key_bytes

    # Corrupting a tail proves the selector never slices or decompresses it.
    damaged = bytearray(split.blob)
    offset = split._entries[1][2]
    _, head_size, tail_size, *_ = _PAGE_SPLIT.unpack_from(split.blob, offset)
    assert tail_size
    damaged[offset + _PAGE_SPLIT.size + head_size] ^= 255
    reopened = ColdStore(bytes(damaged), codec)
    rescanned, damaged_stats = scan_key_coefficients(reopened, 4)
    torch.testing.assert_close(rescanned, coefficients, rtol=0, atol=0)
    assert damaged_stats == stats
