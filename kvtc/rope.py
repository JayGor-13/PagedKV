"""RoPE removal and restoration.

KVTC Section 3.1: "positional embeddings distort the apparent low-rank structure
of keys and should be removed before compression".

RoPE is an orthogonal rotation, so it inverts exactly:
    k' = k*cos + rot(k)*sin
    rot(k') = rot(k)*cos - k*sin          [since rot(rot(x)) = -x]
    => k'*cos - rot(k')*sin = k*(cos^2 + sin^2) = k

The only error is floating point. On bf16 caches expect ~1e-4 relative; on fp32
expect ~1e-7. verify_kvtc.py check 1 asserts this.
"""
import torch


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


def undo_rope(x, cos, sin):
    return x * cos - rotate_half(x) * sin


def get_cos_sin(model, seq_len, device, dtype=torch.float32, position_ids=None):
    """Fetch (cos, sin) shaped (1, 1, seq_len, head_dim) for broadcasting over heads.

    IMPORTANT: HF rotary embeddings cast their output to the dtype of the tensor
    passed in. Passing a bf16 dummy yields bf16 cos/sin tables, which caps the
    RoPE inverse at ~3e-3 relative error. We pass an fp32 dummy so the tables are
    fp32 and the inverse is exact to fp32 (~1e-7), independent of model dtype.
    """
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    inner = model.model if hasattr(model, "model") else model
    rot = inner.rotary_emb
    dummy = torch.zeros(1, seq_len, 1, device=device, dtype=torch.float32)
    cos, sin = rot(dummy, position_ids)
    # Accelerate can place the shared rotary module on a different GPU from the
    # caller. Its hook then returns tables on that module's device, even though
    # ``device`` names the GPU where the attention calculation will run.
    return (cos.unsqueeze(1).to(device=device, dtype=dtype),
            sin.unsqueeze(1).to(device=device, dtype=dtype))


def roundtrip_error(model, seq_len=256, device="cuda"):
    """Max relative error of undo->reapply. Diagnostic for verify_kvtc.py."""
    dt = next(model.parameters()).dtype
    cos, sin = get_cos_sin(model, seq_len, device, torch.float32)
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    k = torch.randn(1, cfg.num_key_value_heads, seq_len, hd,
                    device=device, dtype=torch.float32)
    kr = apply_rope(k, cos, sin)
    back = undo_rope(kr, cos, sin)
    return float((back - k).abs().max() / k.abs().max().clamp_min(1e-9))
    return float((back - k).abs().max() / k.abs().max().clamp_min(1e-9))
