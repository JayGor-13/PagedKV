"""KV cache capture and the cross-layer feature layout (paper Section 3.1).

Feature vector for one token position:
    concat over l layers, h KV heads of the head_dim-vector
    -> p = n_layers * n_kv_heads * head_dim

Appendix B.14 examples:
    Qwen2.5-R1-1.5B  28 * 2 * 128 =  7168
    Qwen2.5-R1-7B    28 * 4 * 128 = 14336
    Llama-3.1-8B     32 * 8 * 128 = 32768
    Llama-3.3-70B    80 * 8 * 128 = 81920

Appendix B.10 shows this cross-layer concatenation is ESSENTIAL, not cosmetic:
per-layer PCA collapses GSM8K from 52.0 to 2.5 at 16x. `pca_layers` in the
config lets you reproduce that collapse as a positive control.
"""
import torch

from .rope import get_cos_sin, undo_rope


def cache_layers(pkv):
    """Extract [(K, V), ...] from a past_key_values object across transformers versions."""
    if hasattr(pkv, "layers"):
        return [(l.keys, l.values) for l in pkv.layers]
    if hasattr(pkv, "key_cache"):
        return list(zip(pkv.key_cache, pkv.value_cache))
    return list(pkv)


def feature_dim(model) -> int:
    c = model.config
    hd = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    return c.num_hidden_layers * c.num_key_value_heads * hd


def layer_groups(n_layers: int, group: int):
    """Split layers into contiguous groups of size `group` (-1 = one group)."""
    if group is None or group < 0 or group >= n_layers:
        return [list(range(n_layers))]
    return [list(range(i, min(i + group, n_layers)))
            for i in range(0, n_layers, group)]


@torch.no_grad()
def capture_document(model, input_ids, pre_rope_keys=True, device="cuda"):
    """Run one document and return (K, V) as (seq_len, p) float32 CPU tensors.

    Keys have RoPE removed when pre_rope_keys=True.
    """
    ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
    ids = ids.to(device)
    out = model(ids, use_cache=True)
    layers = cache_layers(out.past_key_values)
    S = ids.shape[1]
    cos, sin = get_cos_sin(model, S, device, torch.float32)

    ks, vs = [], []
    for (K, V) in layers:
        Kf = K.float()
        if pre_rope_keys:
            Kf = undo_rope(Kf, cos, sin)
        # (1, kv_heads, S, D) -> (S, kv_heads*D)
        ks.append(Kf[0].permute(1, 0, 2).reshape(S, -1))
        vs.append(V[0].float().permute(1, 0, 2).reshape(S, -1))
    K_all = torch.cat(ks, dim=1).cpu()
    V_all = torch.cat(vs, dim=1).cpu()
    del out
    if device == "cuda":
        torch.cuda.empty_cache()
    return K_all, V_all


@torch.no_grad()
def logits_for(model, input_ids, device="cuda"):
    ids = input_ids if input_ids.dim() == 2 else input_ids.unsqueeze(0)
    return model(ids.to(device), use_cache=False).logits.float().cpu()


def split_protected(x, sink, window):
    """Split (S, p) into (compressed_idx, protected_idx) per the paper's policy.

    Protected = first `sink` tokens (attention sinks) + last `window` tokens.
    Paper Section 3.1: s=4, w=128.
    """
    S = x.shape[0]
    if S <= sink + window:
        return torch.empty(0, dtype=torch.long), torch.arange(S)
    comp = torch.arange(sink, S - window)
    prot = torch.cat([torch.arange(0, sink), torch.arange(S - window, S)])
    return comp, prot


def restore_kv_for_model(K, V, model, positions=None, device="cuda",
                         reapply_rope=True):
    """Inverse of capture_document's layout: (S,p) -> per-layer (1,h,S,D) tensors.

    Re-applies RoPE to keys so the result can be fed straight back as a cache.
    """
    from .rope import apply_rope
    c = model.config
    L, H = c.num_hidden_layers, c.num_key_value_heads
    D = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    S = K.shape[0]
    cos, sin = get_cos_sin(model, S, device, torch.float32,
                           position_ids=positions)
    mdtype = next(model.parameters()).dtype
    out = []
    hd = H * D
    for li in range(L):
        layer_device = next(model.model.layers[li].parameters()).device
        k = K[:, li * hd:(li + 1) * hd].to(layer_device).view(S, H, D).permute(1, 0, 2)[None]
        v = V[:, li * hd:(li + 1) * hd].to(layer_device).view(S, H, D).permute(1, 0, 2)[None]
        if reapply_rope:
            k = apply_rope(k, cos.to(layer_device), sin.to(layer_device))
        # the attention kernels require the model's dtype, not fp32
        out.append((k.to(mdtype).contiguous(), v.to(mdtype).contiguous()))
    return out
