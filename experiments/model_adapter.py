"""Qwen2/Llama cache adapter: physically sparse K/V with original RoPE positions.

Adapted from the supplied experiment's cache/pre-fill/sliced-answer routines.
Currently supported: Qwen2 and Llama full-attention models, batch size one.
"""
from dataclasses import dataclass

import torch

from kvtc.capture import cache_layers, restore_kv_for_model
from kvtc.rope import get_cos_sin, undo_rope


def check_model(model):
    if model.config.model_type not in ('qwen2', 'llama') or getattr(model.config, 'use_sliding_window', False):
        raise ValueError('This adapter supports Qwen2/Llama full attention only')
    rope = getattr(model.config, 'rope_parameters', None) or getattr(model.config, 'rope_scaling', None) or {}
    if rope.get('rope_type', rope.get('type', 'default')) not in ('default', 'llama3', 'linear'):
        raise ValueError('RoPE inversion supports default, linear and llama3; scaled-amplitude/dynamic RoPE needs validation')


def make_cache(model, layers):
    from transformers import DynamicCache
    cache = DynamicCache(config=model.config)
    for i, (k, v) in enumerate(layers):
        cache.update(k.clone(), v.clone(), i)
    return cache


def tensor_bytes(layers):
    return sum(t.numel()*t.element_size() for pair in layers for t in pair)


@torch.no_grad()
def prefill(model, ids, chunk=2048, collect_queries=False):
    check_model(model)
    device = next(model.parameters()).device
    cache = make_cache(model, [])
    buffers = [[] for _ in model.model.layers]
    hooks = []
    if collect_queries:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.self_attn.q_proj.register_forward_hook(
                lambda m, inp, out, i=i: buffers[i].append(out[0].detach().cpu())))
    try:
        for start in range(0, len(ids), chunk):
            end = min(len(ids), start+chunk)
            model(ids[start:end][None].to(device), past_key_values=cache,
                  position_ids=torch.arange(start, end, device=device)[None],
                  use_cache=True, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
    return cache_layers(cache), [torch.cat(b, 0) for b in buffers] if collect_queries else None


@torch.no_grad()
def to_features(model, layers):
    n = layers[0][0].shape[2]
    device = layers[0][0].device
    cos, sin = get_cos_sin(model, n, device)
    keys, values = [], []
    for k, v in layers:
        keys.append(undo_rope(k.float(), cos.to(k.device), sin.to(k.device))[0].permute(1, 0, 2).reshape(n, -1).cpu())
        values.append(v[0].float().permute(1, 0, 2).reshape(n, -1).cpu())
    return torch.cat(keys, 1), torch.cat(values, 1)


def slice_layers(layers, positions):
    return [(k.index_select(2, positions.to(k.device)).contiguous(),
             v.index_select(2, positions.to(v.device)).contiguous()) for k, v in layers]


@dataclass
class ActiveCache:
    layers: list
    positions: torch.Tensor
    decoded_pages: tuple
    payload_bytes_read: int


@torch.no_grad()
def recover(model, archive, page_ids, hot_layers, hot_positions):
    """Decode only selected pages, merge with hot entries, let original hot win."""
    recovered = archive.decode_pages(page_ids)
    hot_positions = hot_positions.cpu().long()
    if len(hot_positions) != len(torch.unique(hot_positions)):
        raise ValueError('duplicate hot positions')
    if hot_layers and hot_layers[0][0].shape[2] != len(hot_positions):
        raise ValueError('hot positions/cache length mismatch')
    keep = ~torch.isin(recovered.positions, hot_positions)
    cold_positions = recovered.positions[keep]
    device = next(model.parameters()).device
    if len(cold_positions):
        cold_layers = restore_kv_for_model(recovered.keys[keep], recovered.values[keep], model,
                                          positions=cold_positions[None].to(device), device=device)
    else:
        cold_layers = [(k[:, :, :0], v[:, :, :0]) for k, v in hot_layers]
    positions = torch.cat([hot_positions, cold_positions])
    order = positions.argsort()
    if hot_layers:
        merged = [(torch.cat([hk, ck.to(hk.device)], 2).index_select(2, order.to(hk.device)),
                   torch.cat([hv, cv.to(hv.device)], 2).index_select(2, order.to(hv.device)))
                  for (hk, hv), (ck, cv) in zip(hot_layers, cold_layers)]
    else:
        merged = slice_layers(cold_layers, order)
    return ActiveCache(merged, positions[order], recovered.page_ids, recovered.payload_bytes_read)


@torch.no_grad()
def question_forward(model, layers, original_length, q_ids, collect_queries=False):
    check_model(model)
    device = next(model.parameters()).device
    cache = make_cache(model, layers)
    grabbed = {}
    hooks = []
    if collect_queries:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.self_attn.q_proj.register_forward_hook(
                lambda m, inp, out, i=i: grabbed.__setitem__(i, out[0].detach().float())))
    try:
        result = model(q_ids[None].to(device), past_key_values=cache, use_cache=True,
                       attention_mask=torch.ones(1, cache.get_seq_length()+len(q_ids),
                                                 dtype=torch.long, device=device),
                       position_ids=(original_length+torch.arange(len(q_ids), device=device))[None],
                       logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
    queries = None
    if collect_queries:
        cfg = model.config
        d = getattr(cfg, 'head_dim', cfg.hidden_size//cfg.num_attention_heads)
        queries = [grabbed[i].view(len(q_ids), cfg.num_attention_heads, d).permute(1, 0, 2)
                   for i in range(cfg.num_hidden_layers)]
    return result, queries


@torch.no_grad()
def answer(model, layers, original_length, q_ids, n_new=10):
    if n_new < 1:
        raise ValueError('n_new must be positive')
    result, _ = question_forward(model, layers, original_length, q_ids)
    cache = result.past_key_values
    nxt = result.logits[0, -1].argmax()
    generated = [int(nxt)]
    device = nxt.device
    for step in range(n_new-1):
        result = model(nxt.view(1, 1), past_key_values=cache, use_cache=True,
                       position_ids=torch.tensor([[original_length+len(q_ids)+step]], device=device),
                       logits_to_keep=1)
        nxt = result.logits[0, -1].argmax()
        generated.append(int(nxt))
    return generated


@torch.no_grad()
def answer_profile(model, layers, original_length, q_ids, n_new=10, eos_ids=()):
    """One-shot quality-harness telemetry, including fresh-cache construction.

    Not a repeated serving benchmark. Return first-answer logits for paired KL.
    """
    from .metrics import measured
    if n_new < 1:
        raise ValueError('n_new must be positive')
    device = next(model.parameters()).device
    (result, _), ttft = measured(lambda: question_forward(model, layers, original_length, q_ids), device)
    first_logits = result.logits[0, -1].detach().float().cpu()
    cache = result.past_key_values
    nxt = result.logits[0, -1].argmax()
    generated, latencies = [int(nxt)], []
    for step in range(n_new-1):
        if generated[-1] in eos_ids:
            break
        result, ms = measured(lambda: model(nxt.view(1, 1), past_key_values=cache, use_cache=True,
            position_ids=torch.tensor([[original_length+len(q_ids)+step]], device=device),
            logits_to_keep=1), device)
        latencies.append(ms)
        nxt = result.logits[0, -1].argmax()
        generated.append(int(nxt))
    return generated, first_logits, dict(question_forward_ms=ttft, decode_step_ms=latencies,
        generation_ms=ttft+sum(latencies), generated_tokens=len(generated),
        mean_decode_ms=sum(latencies)/len(latencies) if latencies else None,
        timing_scope='single_quality_harness_sample_not_isolated_serving_benchmark')
