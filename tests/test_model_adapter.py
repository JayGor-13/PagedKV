import pytest
import torch

transformers = pytest.importorskip('transformers')

from kvtc import KVTCCodec, KVTCConfig
from kvtc.cold_store import ColdStore
from kvtc.capture import restore_kv_for_model
from experiments import model_adapter as A
from experiments.selector import scan_key_coefficients


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two CUDA devices')
def test_rope_tables_return_to_requested_gpu_when_rotary_is_sharded():
    from types import SimpleNamespace
    from kvtc.rope import get_cos_sin

    class RotaryOnSecondGpu:
        def __call__(self, dummy, position_ids):
            shape = (1, position_ids.shape[1], 8)
            return (torch.ones(shape, device='cuda:1', dtype=torch.float64),
                    torch.zeros(shape, device='cuda:1', dtype=torch.float64))

    model = SimpleNamespace(model=SimpleNamespace(rotary_emb=RotaryOnSecondGpu()))
    cos, sin = get_cos_sin(model, 16, 'cuda:0', torch.float32)
    assert cos.device == torch.device('cuda:0') and sin.device == torch.device('cuda:0')
    assert cos.dtype == torch.float32 and sin.dtype == torch.float32


@pytest.fixture(scope='module', params=['qwen2', 'llama', 'llama3'])
def sample(request):
    torch.manual_seed(42)
    config_class = transformers.Qwen2Config if request.param == 'qwen2' else transformers.LlamaConfig
    model_class = transformers.Qwen2ForCausalLM if request.param == 'qwen2' else transformers.LlamaForCausalLM
    cfg = config_class(vocab_size=128, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=2048, attention_dropout=0.,
                                   **({'rope_parameters': {'rope_type': 'llama3', 'factor': 4.,
                                       'low_freq_factor': 1., 'high_freq_factor': 4.,
                                       'original_max_position_embeddings': 512}}
                                      if request.param == 'llama3' else {}))
    cfg._attn_implementation = 'eager'
    model = model_class(cfg).eval()
    with torch.no_grad():
        train_layers, _ = A.prefill(model, torch.randint(0, 128, (96,)), chunk=32)
        train = A.to_features(model, train_layers)
        codec = KVTCCodec(KVTCConfig(target_cr=2, pca_rank_cap=32, block_sizes=(4, 16), dp_stride=1,
                                     sink_tokens=2, window_tokens=4, dp_calib_subsample=0), device='cpu')
        codec.calibrate([train[0]], [train[1]], verbose=False)
        layers, _ = A.prefill(model, torch.randint(0, 128, (40,)), chunk=13)
        k, v = A.to_features(model, layers)
        archive = ColdStore.encode(codec, k, v, page_tokens=8)
    return model, codec, archive, layers, k, v


def test_all_pages_vs_full_decode_next_logits(sample):
    model, codec, archive, layers, k, v = sample
    empty = A.slice_layers(layers, torch.empty(0, dtype=torch.long))
    active = A.recover(model, archive, range(archive.page_count), empty, torch.empty(0, dtype=torch.long))
    full = archive.decode_all()
    expected = restore_kv_for_model(full.keys, full.values, model, device='cpu')
    question = torch.tensor([11, 7, 8])
    a, _ = A.question_forward(model, active.layers, 40, question)
    b, _ = A.question_forward(model, expected, 40, question)
    torch.testing.assert_close(a.logits, b.logits, rtol=0, atol=0)
    assert A.answer(model, active.layers, 40, question, 4) == A.answer(model, expected, 40, question, 4)


def test_phase_one_ours_chat_prefill_and_archive_paths(sample, tmp_path):
    from experiments.phase_one_ours import PromptArchive
    from experiments.freekv_protocol import settings
    model, codec, _, _, _, _ = sample
    archive = PromptArchive(model, {}, tmp_path)
    archive.codec = codec
    cfg = dict(settings('longbenchv2', 'ours'), budget=32, sink=32, recent=32)
    ids = [i % 128 for i in range(32)]
    result, metadata = archive.prefill(ids, cfg)
    assert not metadata['prompt_archive_used']
    with torch.no_grad():
        reference = model(torch.tensor([ids]), use_cache=False)
    torch.testing.assert_close(result.logits[:, -1], reference.logits[:, -1], atol=2e-6, rtol=2e-5)
    result, metadata = archive.prefill([i % 128 for i in range(256)], cfg)
    assert metadata['prompt_archive_used'] and metadata['retained_prompt_tokens'] < 192
    assert result.logits.shape[:2] == (1, 1)
    assert metadata['generated_cache_policy'] == 'retain_all_generated_kv'


def test_phase_one_selective_query_capture_is_exact(sample):
    from experiments.phase_one_ours import prefill_scoring_rows
    from experiments.reference_heuristics import eviction_scores
    model = sample[0]
    ids = torch.arange(256) % 128
    dense_layers, dense = A.prefill(model, ids, collect_queries=True)
    sparse_layers, sparse = prefill_scoring_rows(model, ids)
    positions = torch.cat([torch.arange(0, 256, 16), torch.arange(192, 256)]).unique(sorted=True)
    for a, b in zip(dense, sparse):
        torch.testing.assert_close(a[positions], b[positions], atol=0, rtol=0)
        assert len(b.rows) < len(a)
    expected = eviction_scores(model, dense_layers, dense, 256, 'cpu')
    actual = eviction_scores(model, sparse_layers, sparse, 256, 'cpu')
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_sparse_native_matches_masked_reference_with_original_positions(sample):
    model, _, _, layers, _, _ = sample
    positions = torch.tensor([0, 1, 7, 18, 36, 37, 38, 39])
    question = torch.tensor([11, 7, 8])
    sparse = A.slice_layers(layers, positions)
    actual, _ = A.question_forward(model, sparse, 40, question)
    mask = torch.zeros(1, 43, dtype=torch.long)
    mask[0, positions] = 1
    mask[0, 40:] = 1
    with torch.no_grad():
        reference = model(question[None], past_key_values=A.make_cache(model, layers),
                          attention_mask=mask, position_ids=torch.arange(40, 43)[None], logits_to_keep=1)
    torch.testing.assert_close(actual.logits, reference.logits, rtol=1e-5, atol=1e-6)


def test_recall_only_reads_selected_pages_and_hot_overrides(sample, monkeypatch):
    model, _, archive, layers, _, _ = sample
    hot_ids = torch.tensor([0, 1, 8, 36, 37, 38, 39])
    hot = A.slice_layers(layers, hot_ids)
    called = []
    original = archive.decode_pages

    def track(ids):
        called.append(tuple(ids))
        return original(ids)

    monkeypatch.setattr(archive, 'decode_pages', track)
    active = A.recover(model, archive, [1], hot, hot_ids)
    assert called == [(1,)]
    assert len(active.positions) == 14
    assert len(active.positions.unique()) == len(active.positions)
    for (hk, hv), (ak, av) in zip(hot, active.layers):
        idx = torch.searchsorted(active.positions, hot_ids)
        torch.testing.assert_close(ak[:, :, idx], hk, rtol=0, atol=0)
        torch.testing.assert_close(av[:, :, idx], hv, rtol=0, atol=0)


def test_key_scan_reads_only_key_streams_and_matches_stored_quantization(sample, monkeypatch):
    from kvtc.quant import quantize_block
    model, codec, archive, layers, k, v = sample
    calls = []
    original = archive.key_head_symbols

    def track(page, topk):
        calls.append((page, topk))
        return original(page, topk)

    monkeypatch.setattr(archive, 'key_head_symbols', track)
    monkeypatch.setattr(archive, '_page_payloads',
                        lambda *args, **kwargs: pytest.fail('head scan reconstructed a full key payload'))
    coefficients, metrics = scan_key_coefficients(archive, 16)
    assert calls == [(page, 16) for page in range(archive.page_count)]
    assert metrics['values_scanned'] is False
    assert metrics['key_head_rank'] == 16
    assert metrics['key_scan_payload_fraction'] == 1  # This tiny rank has no tail.
    basis = codec.art.key
    for page in range(archive.page_count):
        a, b = page*8, min(len(k), (page+1)*8)
        # Compare against the original codec's quantized coefficients for
        # interior rows, excluding protected positions projected from FP16.
        z = (k[a:b]-basis.mu) @ basis.V
        expected = torch.zeros_like(z)
        for s, e, quant in codec.art.assignments['key'].blocks:
            expected[:, s:e] = quantize_block(z[:, s:e], quant)[0]
        ids = torch.arange(a, b)
        keep = (ids >= 2) & (ids < len(k)-4)
        torch.testing.assert_close(coefficients[ids[keep]], expected[keep, :16], rtol=0, atol=0)
