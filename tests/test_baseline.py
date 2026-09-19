"""Small correctness checks, independent of a model or GPU."""
import itertools

import numpy as np
import pytest
import torch

from kvtc import KVTCCodec, KVTCConfig
from kvtc.dp import assign_precision, assign_precision_bruteforce
from kvtc.pca import fit_pca_gram, project, reconstruct
from kvtc.quant import quantize_block, dequantize_block, used_bits
from kvtc.rope import apply_rope, undo_rope
from kvtc.serialize import pack_codes, unpack_codes


@pytest.mark.parametrize('qtype', ['int2', 'int4', 'fp8'])
@pytest.mark.parametrize('values', [[0., 0., 0., 0.], [1.25]*4,
                                    [0., 1e-10, 2e-10, 3e-10]])
def test_degenerate_quantization_is_finite_and_decodable(qtype, values):
    x = torch.tensor([values], dtype=torch.float32)
    deq, codes, scale, shift = quantize_block(x, qtype)
    assert torch.all(scale > 0)
    assert torch.isfinite(deq).all()
    torch.testing.assert_close(deq, dequantize_block(codes, scale, shift, qtype, 4),
                               rtol=0, atol=0)
    if len(set(values)) == 1:
        # FP8 uses a rounded scale and remains lossy even for a constant group.
        torch.testing.assert_close(deq, x, rtol=1e-3 if qtype == 'fp8' else 0, atol=0)


def assignment_error(x, assignment):
    return sum(float(((x[:, s:e] - quantize_block(x[:, s:e], t)[0])**2).sum())
               for s, e, t in assignment.blocks)


def test_dp_reports_actual_error_and_matches_independent_enumeration():
    x = torch.tensor([[1.2, 2.8, .1, .3], [-1.1, 1.4, -.2, .5],
                      [3.2, 5.1, .1, .8]])
    cfg = KVTCConfig(dp_stride=1, dp_calib_subsample=0,
                     block_sizes=(2,), quant_types=('none', 'int2', 'int4'))
    budget = 76
    result = assign_precision(x, budget, cfg)
    # Enumerate quantizer choices explicitly, without the reference DP helper.
    candidates = []
    for left, right in itertools.product(cfg.quant_types, repeat=2):
        cost = used_bits(left, 2) + used_bits(right, 2)
        if cost <= budget:
            candidates.append(sum(float(((x[:, s:s+2] -
                              quantize_block(x[:, s:s+2], t)[0])**2).sum())
                              for s, t in [(0, left), (2, right)]))
    actual = assignment_error(x, result)
    assert actual == pytest.approx(min(candidates), abs=1e-6)
    assert result.sq_error == pytest.approx(actual, abs=1e-6)
    assert 0 <= result.bits_per_token <= budget
    reference = assign_precision_bruteforce(x, budget, cfg)
    assert reference.sq_error == pytest.approx(min(candidates), abs=1e-6)


def test_skip_transition_does_not_introduce_unconfigured_quantized_blocks():
    x = torch.tensor([[1.234, 5.678, -1.234, -5.678]])
    cfg = KVTCConfig(block_sizes=(2,), dp_stride=1, dp_calib_subsample=0)
    result = assign_precision(x, 1000, cfg)
    assert all(t == 'none' or e - s in cfg.block_sizes for s, e, t in result.blocks)


def test_production_dp_stride_defaults_to_sixteen():
    assert KVTCConfig().dp_stride == 16


def test_partial_stride_is_rejected_instead_of_mischarging_tail():
    with pytest.raises(ValueError, match='divisible'):
        assign_precision(torch.ones(2, 17), 100, KVTCConfig(dp_stride=16))


def test_zero_budget_reports_dropped_energy():
    x = torch.tensor([[1., 2., 3., 4.]])
    cfg = KVTCConfig(dp_stride=1, dp_calib_subsample=0, block_sizes=(2,))
    result = assign_precision(x, 0, cfg)
    assert result.bits_per_token == 0
    assert result.sq_error == pytest.approx(30.)


def test_full_rank_pca_roundtrip_on_unseen_rows():
    rng = torch.Generator().manual_seed(2)
    train = torch.randn(50, 8, generator=rng) + 4
    test = torch.randn(5, 8, generator=rng) + 4
    mu, basis, _ = fit_pca_gram([train[:20], train[20:]], device='cpu')
    torch.testing.assert_close(reconstruct(project(test, mu, basis), mu, basis),
                               test, rtol=1e-5, atol=1e-5)


def test_rope_roundtrip_with_nonconsecutive_original_positions():
    positions = torch.tensor([0., 7., 103.])
    angles = positions[:, None] * torch.tensor([.01, .1])[None, :]
    angles = torch.cat([angles, angles], dim=-1)[None, None, :, :]
    x = torch.arange(12).float().reshape(1, 1, 3, 4)
    torch.testing.assert_close(undo_rope(apply_rope(x, angles.cos(), angles.sin()),
                                       angles.cos(), angles.sin()), x,
                               rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('layout', ['token_major', 'component_major'])
def test_mixed_width_bit_packing(layout):
    widths = np.array([2, 4, 8], dtype=np.uint8)
    codes = np.array([[0, 15, 255], [3, 1, 127], [2, 8, 3]], dtype=np.uint8)
    packed = pack_codes(codes, widths, layout)
    assert len(packed) == 6  # 42 payload bits, padded to six bytes.
    np.testing.assert_array_equal(unpack_codes(packed, 3, widths, layout), codes)


@pytest.mark.parametrize('length', [3, 5, 17])
@pytest.mark.parametrize('layout', ['token_major', 'component_major'])
def test_serialized_codec_matches_quantizer_and_counts_all_bytes(length, layout):
    rng = torch.Generator().manual_seed(7)
    train = torch.randn(48, 8, generator=rng)
    cfg = KVTCConfig(target_cr=2, pca_rank_cap=8, dp_stride=1,
                     block_sizes=(2, 4, 8), dp_calib_subsample=0,
                     sink_tokens=1, window_tokens=2, layout=layout)
    codec = KVTCCodec(cfg, device='cpu')
    codec.calibrate([train], [train * .7], verbose=False)
    k = torch.randn(length, 8, generator=rng)
    v = torch.randn(length, 8, generator=rng)
    payloads = codec.compress(k, v)
    recovered = codec.decompress(payloads)
    for name, x, decoded in zip(('key', 'value'), (k, v), recovered):
        sym = codec.encode_symbols(x, name)
        # Independent expected numerical reconstruction, bypassing serialization.
        basis = getattr(codec.art, name)
        expected = x.half().float()
        ids = sym['comp_idx']
        if len(ids):
            z = project(x[ids], basis.mu, basis.V)
            zhat = torch.zeros_like(z)
            for s, e, t in codec.art.assignments[name].blocks:
                zhat[:, s:e] = quantize_block(z[:, s:e], t)[0]
            expected[ids] = reconstruct(zhat, basis.mu, basis.V)
        torch.testing.assert_close(decoded, expected, rtol=1e-6, atol=1e-6)
        payload = payloads[name]
        assert payload.nbytes() == sum(payload.stored.values())
    measured = codec.compression_ratio(payloads, length, 8)
    assert measured['total_bytes'] == sum(p.nbytes() for p in payloads.values())
