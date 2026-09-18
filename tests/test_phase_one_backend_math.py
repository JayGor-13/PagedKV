"""Tiny CPU adapter checks under TF 4.45.2, replacing CUDA attention with SDPA.

These validate cache/position integration, not native CUDA kernels or H200 speed.
Run separately with .envs/phase-test/Scripts/python -m pytest on Windows, or the
baseline environment on Linux. No pretrained weights are downloaded.
"""
import copy
import sys
import types
import importlib.machinery
import pytest
import torch
import transformers

pytestmark = pytest.mark.skipif(transformers.__version__ != '4.45.2', reason='isolated TF 4.45.2 adapter test')


@pytest.fixture
def attention_shim(monkeypatch):
    def flash(q, k, v, dropout_p=0., softmax_scale=None, causal=False, **kwargs):
        if q.shape[2] != k.shape[2]:
            k = k.repeat_interleave(q.shape[2]//k.shape[2], dim=2)
            v = v.repeat_interleave(q.shape[2]//v.shape[2], dim=2)
        mask = None
        if causal:
            mask = torch.arange(k.shape[1])[None, :] <= torch.arange(q.shape[1])[:, None] + k.shape[1]-q.shape[1]
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                               attn_mask=mask, scale=softmax_scale).transpose(1, 2)
    for name in ('flash_attn', 'flash_attn.bert_padding', 'flash_attn.flash_attn_interface', 'flashinfer'):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules['flash_attn'].flash_attn_func = flash
    sys.modules['flash_attn.flash_attn_interface'].flash_attn_func = flash
    sys.modules['flash_attn'].flash_attn_varlen_func = lambda *a, **k: (_ for _ in ()).throw(AssertionError('padding not expected'))
    for name in ('index_first_axis', 'pad_input', 'unpad_input'):
        setattr(sys.modules['flash_attn.bert_padding'], name, lambda *a, **k: None)
    monkeypatch.setenv('INPLACE_ROPE_OFF', '1')


@pytest.mark.parametrize('architecture', ['llama', 'qwen2'])
@pytest.mark.parametrize('method', ['full', 'quest', 'arkvale', 'freekv', 'snapkv', 'h2o'])
def test_tiny_adapter_decode_and_reset(attention_shim, architecture, method):
    from experiments.phase_one_backends import patch_model
    from experiments.phase_one_worker import generate
    from experiments.freekv_protocol import settings
    cls = transformers.LlamaConfig if architecture == 'llama' else transformers.Qwen2Config
    config = cls(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                 attention_dropout=0., eos_token_id=96)
    config._attn_implementation = 'eager'
    torch.manual_seed(5)
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    cfg = dict(settings('longbenchv2', method), budget=32, sink=32, recent=32, max_new_tokens=4)
    model, updater = patch_model(model, method, cfg)
    example = dict(id='same', token_ids=[i % 95 for i in range(160)])
    first, _ = generate(model, updater, method, example, cfg, 42, [], [])
    second, _ = generate(model, updater, method, example, cfg, 42, [], [])
    assert first == second and len(first) == 4
    assert all(0 <= x < 97 for x in first)


@pytest.mark.parametrize('architecture', ['llama', 'qwen2', 'llama3'])
def test_full_cache_fix_matches_unmodified_hf(attention_shim, architecture):
    from experiments.phase_one_backends import patch_model
    from experiments.freekv_protocol import settings
    cls = transformers.Qwen2Config if architecture == 'qwen2' else transformers.LlamaConfig
    options = dict(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                   attention_dropout=0.)
    if architecture == 'llama3':
        options['rope_scaling'] = dict(rope_type='llama3', factor=8., low_freq_factor=1., high_freq_factor=4., original_max_position_embeddings=64)
    config = cls(**options)
    config._attn_implementation = 'eager'
    reference = transformers.AutoModelForCausalLM.from_config(config).eval()
    model, _ = patch_model(copy.deepcopy(reference), 'full', settings('longbenchv2', 'full'))
    ids = torch.arange(80).view(1, -1)
    with torch.no_grad():
        expected = reference(ids, use_cache=True)
        actual = model(ids, use_cache=True)
        torch.testing.assert_close(actual.logits[:, -1], expected.logits[:, -1], atol=2e-6, rtol=2e-5)
        next_id = torch.tensor([[85]])
        expected = reference(next_id, use_cache=True, past_key_values=expected.past_key_values)
        actual = model(next_id, use_cache=True, past_key_values=actual.past_key_values)
        torch.testing.assert_close(actual.logits[:, -1], expected.logits[:, -1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('architecture', ['llama', 'qwen2'])
def test_rocket_decode_resets_upstream_global_position(attention_shim, monkeypatch, architecture):
    from experiments.phase_one_backends import patch_model
    from experiments.phase_one_worker import generate
    from experiments.freekv_protocol import settings
    original_arange = torch.arange
    def cpu_arange(*args, **kwargs):
        if str(kwargs.get('device', '')).startswith('cuda'):
            kwargs['device'] = 'cpu'
        return original_arange(*args, **kwargs)
    monkeypatch.setattr(torch, 'arange', cpu_arange)
    cls = transformers.LlamaConfig if architecture == 'llama' else transformers.Qwen2Config
    config = cls(vocab_size=97, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                     max_position_embeddings=512)
    config._attn_implementation = 'eager'
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    cfg = dict(settings('longbenchv2', 'rocketkv'), budget=32, sink=32, recent=32, max_new_tokens=4)
    model, updater = patch_model(model, 'rocketkv', cfg)
    example = dict(id='rocket', token_ids=[i % 95 for i in range(160)])
    first, _ = generate(model, updater, 'rocketkv', example, cfg, 42, [], [])
    second, _ = generate(model, updater, 'rocketkv', example, cfg, 42, [], [])
    assert first == second and len(first) == 4


def test_chunked_h2o_selects_same_cache_as_dense_scoring(attention_shim):
    from experiments.benchmark_state import ROOT
    sys.path.insert(0, str(ROOT / 'external/KVCache-Factory'))
    from pyramidkv.pyramidkv_utils import H2OKVCluster, _reduce_group_scores, _select_topk_kv
    from experiments.phase_one_backends import chunked_h2o
    torch.manual_seed(25)
    q = torch.randn(1, 4, 300, 8)
    k, v = torch.randn(1, 2, 300, 8), torch.randn(1, 2, 300, 8)
    logits = q @ k.repeat_interleave(2, dim=1).transpose(-1, -2) / 8**.5
    mask = torch.arange(300)[None, :] > torch.arange(300)[:, None]
    logits.masked_fill_(mask, torch.finfo(logits.dtype).min)
    scores = _reduce_group_scores(logits.softmax(-1)[..., :-32].sum(-2), 2, 'mean')
    expected = _select_topk_kv(k, v, scores, 64, 32, None)
    actual = chunked_h2o(H2OKVCluster(window_size=32, max_capacity_prompt=96), k, q, v, 2)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
