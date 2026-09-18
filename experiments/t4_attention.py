"""Explicit single-sequence SDPA replacement for the pinned upstream FA2 API.

This is a portable correctness path, not FlashAttention or a performance port.
Install only in a dedicated T4 baseline worker, after importing Transformers.
Unsupported padding/quantization/attention options fail rather than being ignored.
"""
import importlib.machinery
import os
import sys
import types

import torch
import torch.nn.functional as F


def sdpa_attention(q, k, v, dropout_p=0., softmax_scale=None, causal=False,
                   window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                   return_attn_probs=False, **kwargs):
    if kwargs or window_size != (-1, -1) or alibi_slopes is not None or return_attn_probs:
        raise NotImplementedError('T4 adapter supports plain unpadded attention only')
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError('Expected batch-one [B, tokens, heads, head_dim] Q/K/V')
    if q.shape[-1] != k.shape[-1] or q.shape[2] % k.shape[2]:
        raise ValueError('Incompatible head dimensions or GQA grouping')
    if q.shape[2] != k.shape[2]:
        groups = q.shape[2] // k.shape[2]
        k, v = k.repeat_interleave(groups, 2), v.repeat_interleave(groups, 2)
    nq, nk = q.shape[1], k.shape[1]
    mask = None
    # FA2 uses bottom-right alignment. SDPA is_causal uses top-left alignment
    # for non-square inputs, which would incorrectly mask cached decode tokens.
    square_causal = causal and nq == nk
    if causal and nq != nk:
        mask = torch.arange(nk, device=q.device)[None, :] <= (
            torch.arange(nq, device=q.device)[:, None] + nk - nq)
    result = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, is_causal=square_causal, dropout_p=dropout_p, scale=softmax_scale)
    return result.transpose(1, 2).contiguous()


def unsupported(*args, **kwargs):
    raise NotImplementedError('Padding, variable-length kernels and FlashInfer are disabled in T4 mode')


def install_sdpa_bridge():
    if 'transformers' not in sys.modules:
        raise RuntimeError('Import Transformers before installing the T4 bridge')
    os.environ['INPLACE_ROPE_OFF'] = '1'
    for name in ('flash_attn', 'flash_attn.bert_padding', 'flash_attn.flash_attn_interface', 'flashinfer'):
        previous = sys.modules.get(name)
        if previous is not None and not getattr(previous, '_pagedkv_t4', False):
            raise RuntimeError(f'{name} was already imported; use an isolated T4 environment')
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        module._pagedkv_t4 = True
        module.__getattr__ = lambda name: unsupported if not name.startswith('__') else None
        sys.modules[name] = module
    sys.modules['flash_attn'].flash_attn_func = sdpa_attention
    sys.modules['flash_attn'].flash_attn_varlen_func = unsupported
    sys.modules['flash_attn.flash_attn_interface'].flash_attn_func = sdpa_attention
    for name in ('index_first_axis', 'pad_input', 'unpad_input'):
        setattr(sys.modules['flash_attn.bert_padding'], name, unsupported)
