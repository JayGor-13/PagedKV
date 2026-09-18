"""Exercise calibration, actual byte serialization and reconstruction on CPU.

This uses synthetic correlated vectors, not a language model or QA benchmark.
"""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import torch

from kvtc import KVTCCodec, KVTCConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/smoke.json'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(17)
    p = 64
    key_map = torch.randn(6, p, generator=rng)
    value_map = torch.randn(6, p, generator=rng)

    def sample(n, mapping):
        return (torch.randn(n, 6, generator=rng) @ mapping
                + .02 * torch.randn(n, p, generator=rng))

    cfg = KVTCConfig(target_cr=8, pca_rank_cap=16, block_sizes=(4, 16),
                     dp_stride=1, dp_calib_subsample=0,
                     sink_tokens=4, window_tokens=16, seed=17)
    codec = KVTCCodec(cfg, device='cpu')
    codec.calibrate([sample(256, key_map)], [sample(256, value_map)], verbose=False)
    key, value = sample(256, key_map), sample(256, value_map)
    payloads = codec.compress(key, value)
    decoded = codec.decompress(payloads)
    if not all(torch.isfinite(x).all() for x in decoded):
        raise RuntimeError('The reconstructed cache contains nonfinite values')
    relative_errors = {
        name: float(((a - b)**2).sum() / a.square().sum().clamp_min(1e-12))
        for name, a, b in zip(('key', 'value'), (key, value), decoded)
    }
    metrics = codec.compression_ratio(payloads, len(key), p)
    report = {
        'run_kind': 'synthetic_correctness_only',
        'is_llm_quality_result': False,
        'device': 'cpu',
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'seed': cfg.seed,
        'calibration_rows_per_cache_type': 256,
        'held_out_rows': len(key),
        'features_per_cache_type': p,
        'target_cr': cfg.target_cr,
        'relative_mse': relative_errors,
        'stored_cache': metrics,
        'bytes_by_part': {name: pl.stored for name, pl in payloads.items()},
        'shared_basis_resident_tensor_bytes': sum(
            tensor.numel()*tensor.element_size()
            for basis in (codec.art.key, codec.art.value)
            for tensor in (basis.mu, basis.V, basis.evals)),
        'scope': 'No model accuracy, hot/cold retrieval, GPU timing or novelty claim.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('Synthetic codec round trip completed; this is not an LLM benchmark.')
    print(f'Actual stored cache: {metrics["total_bytes"]:,} bytes')
    print(f'Overall cache compression ratio: {metrics["cr_end_to_end"]:.3f}x')
    print(f'Relative MSE: key={relative_errors["key"]:.6f}, '
          f'value={relative_errors["value"]:.6f}')
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
