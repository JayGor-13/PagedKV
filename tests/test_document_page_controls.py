import pytest

from scripts.audit_page_controls import audit, make_sample


@pytest.fixture(scope='module', params=[(20260915, 1024), (20260916, 1019)])
def sample(request):
    return make_sample(*request.param)


@pytest.mark.parametrize('page_tokens', [64, 128, 256])
def test_shared_basis_global_4_128_protection_and_exact_page_decode(sample, page_tokens):
    codec, keys, values = sample
    result = audit(codec, keys, values, page_tokens)
    assert result['protected_tokens_per_cache_type'] == 132
    assert result['per_page_calibration_calls'] == 0
    assert result['basis_tensor_bytes_stored_in_archive'] == 0
    assert all(result['isolated_vs_same_archive_full_exact'].values())
    assert all(result['quantized_codes_widths_scales_shifts_vs_monolithic_exact'].values())
