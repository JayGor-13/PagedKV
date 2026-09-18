"""GPU-count policy. Model sharding and independent jobs are different layouts."""
import math


def choose_layout(method, model_spec, gpu_count, memory_gib, context, batch=1):
    if gpu_count < 1 or memory_gib <= 0 or context < 1 or batch < 1:
        raise ValueError('positive GPU count, memory, context and batch required')
    weights = 2 * model_spec['parameter_count_approx']
    kv = 4 * model_spec['layers'] * model_spec['kv_heads'] * model_spec['head_dim'] * context * batch
    # Conservative planning estimate only. Our quality runner keeps multiple caches.
    reserve = (12 if method == 'ours' else 6) * 2**30
    need = weights + kv * (4 if method == 'ours' else 1) + reserve
    usable = memory_gib * 2**30 * .9
    count = max(1, math.ceil(need / usable))
    if count > gpu_count:
        return dict(status='insufficient_memory', gpus=count, estimated_bytes=need,
                    reason=f'estimated {need / 2**30:.1f} GiB exceeds {gpu_count} GPU planning capacity; no automatic weight quantization or CPU offload')
    if count > 1 and method in ('arkvale', 'kivi'):
        return dict(status='unsupported', gpus=count, estimated_bytes=need,
                    reason='official cache bridge has no validated model-sharded execution')
    return dict(status='ready', gpus=count, estimated_bytes=need,
                layout='single' if count == 1 else 'model_sharded')


def model_load_kwargs(gpus, memory_gib, device='cuda', modern=True):
    if gpus < 1:
        raise ValueError('gpus must be positive')
    if device == 'cpu' or gpus == 1:
        return {}
    import torch
    if torch.cuda.device_count() < gpus:
        raise ValueError('fewer visible CUDA devices than --gpus')
    return dict(device_map='balanced', max_memory={i: f'{memory_gib}GiB' for i in range(gpus)})


def assert_gpu_only(model):
    mapping = getattr(model, 'hf_device_map', {})
    if any(str(d) in ('cpu', 'disk') for d in mapping.values()):
        raise RuntimeError('model does not fit requested GPUs; refusing implicit CPU/disk offload')
