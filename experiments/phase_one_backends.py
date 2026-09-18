"""Isolated quality adapters. Provenance labels are part of every result."""
import math
import os
import sys
import types
from .benchmark_state import ROOT

LABELS = {
    'full': 'Full cache (FreeKV accuracy implementation, FlashAttention-2)',
    'quest': 'Quest (FreeKV accuracy implementation)',
    'arkvale': 'ArkVale (FreeKV accuracy implementation)',
    'freekv': 'FreeKV (upstream accuracy implementation)',
    'h2o': 'H2O (KVCache-Factory prefill compression, chunked scoring)',
    'snapkv': 'SnapKV (KVCache-Factory implementation)',
    'rocketkv': 'RocketKV (upstream HF math; local Qwen/device bridge)',
    'ours': 'Ours (KVTC prompt archive + query scan; generated KV retained)',
}


def chunked_h2o(cluster, key_states, query_states, value_states, groups):
    """Same Factory full-prefill heavy-hitter score, without an N x N tensor."""
    import torch
    from pyramidkv.pyramidkv_utils import _reduce_group_scores, _select_topk_kv
    n = query_states.shape[-2]
    if n < cluster.max_capacity_prompt:
        return key_states, value_states
    keys = key_states.repeat_interleave(groups, dim=1)
    scores = torch.zeros(*query_states.shape[:2], n-cluster.window_size,
                         dtype=torch.float32, device=query_states.device)
    positions = torch.arange(n, device=query_states.device)
    for start in range(0, n, 128):
        q = query_states[:, :, start:start+128]
        logits = torch.matmul(q, keys.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        mask = positions[None, :] > positions[start:start+q.shape[-2], None]
        logits.masked_fill_(mask, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        scores.add_(probabilities[..., :-cluster.window_size].sum(dim=-2, dtype=torch.float32))
    scores = _reduce_group_scores(scores.to(query_states.dtype), groups, cluster.gqa_score_agg)
    return _select_topk_kv(key_states, value_states, scores, cluster.max_capacity_prompt-cluster.window_size,
                          cluster.window_size, cluster.merge)


def source_for(method):
    return 'factory' if method in ('h2o', 'snapkv') else 'rocketkv' if method == 'rocketkv' else 'freekv'


def support_reason(method, architecture, gpus):
    if architecture not in ('llama', 'qwen2'):
        return 'This comparison supports Llama and Qwen2 architectures only.'
    return None


def rocket_bridge(model):
    """Repair upstream Qwen symbol/rotary API and make rotary tables device-local."""
    import importlib
    import torch
    from transformers import Qwen2ForCausalLM
    module = importlib.import_module('inf_llm.utils.patch')
    from inf_llm.attention.rope import RotaryEmbeddingESM
    module.Qwen2ForCausalLM = Qwen2ForCausalLM

    class DeviceRotary(RotaryEmbeddingESM):
        def _update_cos_sin_tables(self, x, seq_dim, seq_len=0):
            n = seq_len or x.size(seq_dim)
            tables = getattr(self, '_device_tables', {})
            key = (str(x.device), x.dim())
            cached_n, cos, sin = tables.get(key, (0, None, None))
            if n > cached_n:
                inv = self.inv_freq.to(x.device)
                positions = torch.arange(n, device=x.device, dtype=inv.dtype)
                freqs = torch.outer(positions * self.distance_scale, inv)
                emb = torch.cat((freqs, freqs), dim=-1)
                shape = [1] * (x.dim()-2) + list(emb.shape)
                cos, sin = emb.cos().view(shape), emb.sin().view(shape)
                tables[key] = (n, cos, sin)
                self._device_tables = tables
            return cos, sin

    module.RotaryEmbeddingESM = DeviceRotary
    if os.environ.get('PAGEDKV_ROCKET_FP32_ATTENTION') == '1':
        # The pinned RocketKV PyTorch path computes retrieval scores in the
        # model dtype. FP16 dot products can overflow on T4 even when Q/K are
        # finite. Keep weights, caches, and layer outputs FP16 while performing
        # only score, softmax, and value accumulation in FP32.
        import inspect
        import textwrap
        attention = importlib.import_module('inf_llm.attention')
        rocket = importlib.import_module('inf_llm.attention.rocket')
        source = textwrap.dedent(inspect.getsource(rocket.rocket_forward))
        replacements = {
            'score = torch.matmul(h_q_observe, h_k2.transpose(-1, -2)) / math.sqrt(dim_head)':
                'score = torch.matmul(h_q_observe.float(), h_k2.transpose(-1, -2).float()) / math.sqrt(dim_head)',
            'return torch.softmax(x / divscale, dim=dim)':
                'return _pagedkv_fp32_softmax(x, divscale, dim)',
            'QK_hat = Q_hat @ K_hat.transpose(-1, -2)':
                'QK_hat = Q_hat.float() @ K_hat.transpose(-1, -2).float()',
            'QK = Q @ _gather(K, -2, iKV).transpose(-1, -2)':
                'QK = Q.float() @ _gather(K, -2, iKV).transpose(-1, -2).float()',
            'o = s @ _gather(V, -2, iKV)':
                'o = (s @ _gather(V, -2, iKV).float()).to(V.dtype)',
        }
        for old, new in replacements.items():
            if source.count(old) != 1:
                raise RuntimeError('Pinned RocketKV source changed; cannot install the audited T4 FP32 score patch')
            source = source.replace(old, new)
        def stable_softmax(x, divscale, dim):
            scale = torch.as_tensor(divscale, device=x.device, dtype=torch.float32)
            scale = torch.nan_to_num(scale, nan=1., posinf=torch.finfo(torch.float32).max,
                                     neginf=1.).clamp_min(torch.finfo(torch.float32).tiny)
            return torch.softmax(x.float() / scale, dim=dim)
        rocket._pagedkv_fp32_softmax = stable_softmax
        namespace = {}
        exec(compile(source, str(rocket.__file__) + ':pagedkv-fp32', 'exec'), rocket.__dict__, namespace)
        attention.ATTN_FORWRAD['rocket'] = namespace['rocket_forward']
    if model.config.model_type == 'qwen2':
        rotary = model.model.layers[0].self_attn.rotary_emb
        rotary.base = model.config.rope_theta
        rotary.dim = model.config.hidden_size // model.config.num_attention_heads
    return module.patch_hf


def load_model(manifest, method, config, gpus, memory_gib):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from .gpu_layout import model_load_kwargs, assert_gpu_only
    expected = '5.16.1' if method == 'ours' else '4.45.2'
    if transformers.__version__ != expected:
        raise RuntimeError(f'{method} requires its isolated Transformers {expected} environment')
    kwargs = model_load_kwargs(gpus, memory_gib)
    model = AutoModelForCausalLM.from_pretrained(
        manifest['model'], revision=manifest['revision'], torch_dtype=torch.bfloat16,
        attn_implementation='sdpa' if method == 'ours' else 'eager', **kwargs).eval()
    if gpus == 1:
        model.to('cuda:0')
    assert_gpu_only(model)
    return patch_model(model, method, config)


def patch_model(model, method, config):
    """Attach upstream math separately from weight loading for tiny-model tests."""
    device_map = getattr(model, 'hf_device_map', None)
    if device_map:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(model, recurse=True)
    updater = None
    if method in ('full', 'quest', 'arkvale', 'freekv'):
        os.environ['INPLACE_ROPE_OFF'] = '1'  # upstream multi-device rotary path
        sys.path.insert(0, str(ROOT / 'external/FreeKV/accuracy'))
        from kvc.patch.tuple_kv_cache import enable_tuple_kv_cache
        from kvc.patch import enable_dyn_attention, QuestUpdater, SpecRetUpdater
        if method == 'full':
            enable_tuple_kv_cache(model)
            # Pinned upstream only assigns the returned cache in its kv8 branch.
            # Capture the exact K/V passed to FlashAttention and return it for
            # BF16 too, without changing attention computation.
            for layer in model.model.layers:
                attn = layer.self_attn
                original_flash, original_forward = attn._flash_attention_forward, attn.forward
                def flash_with_cache(self, q, k, v, *args, _flash=original_flash, **kwargs):
                    self._phase_exact_cache = (k, v)
                    return _flash(q, k, v, *args, **kwargs)
                def forward_with_cache(self, *args, _forward=original_forward, **kwargs):
                    try:
                        result = _forward(*args, **kwargs)
                        return result[0], result[1], self._phase_exact_cache if kwargs.get('use_cache') else None
                    finally:
                        if hasattr(self, '_phase_exact_cache'):
                            delattr(self, '_phase_exact_cache')
                attn._flash_attention_forward = types.MethodType(flash_with_cache, attn)
                attn.forward = types.MethodType(forward_with_cache, attn)
        else:
            import numpy as np
            shape = (model.config.num_hidden_layers, model.config.num_key_value_heads)
            cfg = dict(kv8=False, sparsity=1., sink=config['sink'], recent=config['recent'],
                       skip_layer=config['skip_layer'], page_rep='quest', budget=config['budget'],
                       page_size=config['page_size'], GQA_policy=config['GQA_policy'],
                       spec_ret_steps=config['spec_ret_steps'], llb=0,
                       correct_sim=config['correct_sim'], corr_group='avg')
            enable_dyn_attention(model, np.zeros(shape), np.ones(shape), cfg['sink'], cfg['recent'],
                                 {'quest': 'quest', 'arkvale': 'arkv', 'freekv': 'spec_ret'}[method], cfg)
            for layer in model.model.layers:
                attn = layer.self_attn
                # llb=0 never reads the previous layer; remove the registered alias
                # before redispatch so Accelerate cannot place it on another GPU.
                if hasattr(attn, 'last_layer_attn'):
                    delattr(attn, 'last_layer_attn')
                for name in ('full_attention_heads', 'dyn_attention_heads'):
                    value = getattr(attn, name, None)
                    if value is not None:
                        setattr(attn, name, value.to(attn.q_proj.weight.device))
            updater = SpecRetUpdater(model) if method == 'freekv' else QuestUpdater(model)
    elif method in ('h2o', 'snapkv'):
        sys.path.insert(0, str(ROOT / 'external/KVCache-Factory'))
        from pyramidkv.llama_model import llama_flash_attn2_forward_H2O, llama_flash_attn2_forward_SnapKV
        if method == 'h2o':
            from pyramidkv.pyramidkv_utils import H2OKVCluster
            H2OKVCluster._update_kv_kv_head = chunked_h2o
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        fn = llama_flash_attn2_forward_H2O if method == 'h2o' else llama_flash_attn2_forward_SnapKV
        model.config._attn_implementation = 'flash_attention_2'
        for layer in model.model.layers:
            attn = layer.self_attn
            attn.config.max_capacity_prompt = config['budget'] + config['sink'] + config['recent']
            attn.config.window_size = 32
            attn.config.kernel_size = 5
            attn.config.pooling = 'avgpool'
            attn.config.merge = None
            attn.config.gqa_score_agg = 'mean'
            attn.config.kv_cache_granularity = 'kv_head'
            # Both architectures use the same q/k/v projection layout. Supply the
            # Llama rotary interface expected by Factory, with the original config.
            attn.rotary_emb = LlamaRotaryEmbedding(config=model.config).to(attn.q_proj.weight.device)
            attn._flash_attn_uses_top_left_mask = False
            attn.forward = types.MethodType(fn, attn)
        # Avoid materializing [prompt_tokens, vocabulary] logits.
        model.lm_head.register_forward_pre_hook(lambda module, inputs: (inputs[0][:, -1:, :],))
    elif method == 'rocketkv':
        sys.path.insert(0, str(ROOT / 'external/RocketKV/pipeline/inf_stream_llm'))
        model.lm_head.register_forward_pre_hook(lambda module, inputs: (inputs[0][:, -1:, :],))
    if device_map:
        from accelerate import dispatch_model
        model = dispatch_model(model, device_map=device_map, force_hooks=True, skip_keys=['past_key_values'])
    return model, updater


def reset(model, updater, method, ids, config):
    import torch
    if updater is not None:
        updater.reset(ids)
        if method == 'freekv':
            for layer in model.model.layers:
                attn = layer.self_attn
                attn.q_ptr = 0
                attn.q_cache.zero_()
                attn.num_correct_kv_heads = torch.zeros(1, device=attn.q_proj.weight.device, dtype=torch.int64)
    if method in ('h2o', 'snapkv'):
        for layer in model.model.layers:
            layer.self_attn.kv_seq_len = 0
            if hasattr(layer.self_attn, 'kv_cluster'):
                delattr(layer.self_attn, 'kv_cluster')
    if method == 'rocketkv':
        device_map = getattr(model, 'hf_device_map', None)
        if device_map:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(model, recurse=True)
        patch_hf = rocket_bridge(model)
        length = ids.shape[-1] + config['max_new_tokens']
        budget = config['budget'] + config['sink'] + config['recent']
        ratio = max(1., length / budget)
        exponent = min(.2 + math.log2(ratio) * .06, .8)
        capacity = max(int(length / ratio**exponent), min(2 * config['max_new_tokens'], length))
        patch_hf(model, 'rocket', attn_kwargs={}, fattn=True, topk=budget // 2,
                 compression_ratio=max(1., capacity / budget),
                 prompt_budget=max(32, capacity - config['max_new_tokens']),
                 window_size=32, kernel_size=63, skip_layers=0)
        if device_map:
            from accelerate import dispatch_model
            dispatch_model(model, device_map=device_map, force_hooks=True, skip_keys=['past_key_values'])
