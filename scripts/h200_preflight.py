"""Fail-fast H200 readiness audit for the phase-one benchmark suite.

Run this from the pinned phase-baselines environment after setup. The report is
safe to archive with results: it records hardware and package versions, never a
Hugging Face token.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.benchmark_state import ROOT, atomic_json


EXPECTED = {
    'phase-baselines': {'torch': '2.5.1', 'transformers': '4.45.2', 'accelerate': '0.34.2',
                        'flash-attn': '2.6.3', 'flashinfer-python': '0.2.4'},
    'phase-ours': {'torch': '2.8.0', 'transformers': '5.16.1'},
    'phase-judge': {'vllm': '0.10.2', 'transformers': '4.55.2'},
}


def check(name, status, detail):
    if status not in ('pass', 'warn', 'fail'):
        raise ValueError('invalid check status')
    return {'name': name, 'status': status, 'detail': str(detail)}


def evaluate_gpu_inventory(devices, requested, minimum_memory_gib):
    checks = []
    if len(devices) < requested:
        checks.append(check('gpu_count', 'fail', f'{len(devices)} visible; {requested} required'))
        return checks
    checks.append(check('gpu_count', 'pass', f'{len(devices)} visible; scheduler will use {requested}'))
    selected = devices[:requested]
    wrong = [d['name'] for d in selected if 'H200' not in d['name'].upper()]
    checks.append(check('gpu_model', 'fail' if wrong else 'pass',
                        f'non-H200 devices: {wrong}' if wrong else ', '.join(d['name'] for d in selected)))
    small = [f"GPU {d['index']}: {d['total_memory_gib']:.1f} GiB" for d in selected
             if d['total_memory_gib'] < minimum_memory_gib]
    checks.append(check('gpu_memory', 'fail' if small else 'pass',
                        '; '.join(small) if small else f'all selected GPUs have at least {minimum_memory_gib} GiB'))
    bad_capability = [d['index'] for d in selected if tuple(d['compute_capability']) < (9, 0)]
    checks.append(check('compute_capability', 'fail' if bad_capability else 'pass',
                        f'below SM90: {bad_capability}' if bad_capability else 'all selected GPUs are SM90 or newer'))
    unsupported = [d['index'] for d in selected if not d['bf16_supported']]
    checks.append(check('bf16', 'fail' if unsupported else 'pass',
                        f'BF16 unavailable on: {unsupported}' if unsupported else 'BF16 supported on every selected GPU'))
    return checks


def evaluate_host_capacity(free_disk_gib, ram_gib, minimum_disk_gib, recommended_ram_gib):
    return [
        check('free_disk', 'fail' if free_disk_gib < minimum_disk_gib else 'pass',
              f'{free_disk_gib:.1f} GiB free; minimum {minimum_disk_gib} GiB'),
        check('host_ram', 'warn' if ram_gib < recommended_ram_gib else 'pass',
              f'{ram_gib:.1f} GiB total; {recommended_ram_gib} GiB recommended for the 72B ours diagnostic path'),
    ]


def environment_versions(root):
    inventories, checks = {}, []
    for name, expected in EXPECTED.items():
        python = root / '.envs' / name / 'bin' / 'python'
        if not python.exists():
            checks.append(check(f'environment_{name}', 'fail', f'missing {python}'))
            continue
        packages = json.dumps(list(expected))
        code = ("import importlib.metadata,json; "
                f"print(json.dumps({{p:importlib.metadata.version(p) for p in {packages}}}))")
        try:
            versions = json.loads(subprocess.check_output([str(python), '-c', code], text=True).strip())
            inventories[name] = {'python': str(python.resolve()), 'packages': versions}
            mismatches = {p: {'expected': v, 'actual': versions.get(p)} for p, v in expected.items()
                          if versions.get(p) != v}
            checks.append(check(f'environment_{name}', 'fail' if mismatches else 'pass',
                                json.dumps(mismatches, sort_keys=True) if mismatches else json.dumps(versions, sort_keys=True)))
        except Exception as error:
            checks.append(check(f'environment_{name}', 'fail', error))
    return inventories, checks


def environment_runtime_checks(root):
    commands = {
        'phase-baselines': ('import torch,flash_attn,flashinfer; '
                            'assert torch.cuda.is_available(); '
                            'x=torch.ones((64,64),device="cuda",dtype=torch.bfloat16); '
                            'assert torch.isfinite(x@x).all(); print("CUDA extensions and BF16 OK")'),
        'phase-ours': ('import torch,transformers; assert torch.cuda.is_available(); '
                       'x=torch.ones((64,64),device="cuda",dtype=torch.bfloat16); '
                       'assert torch.isfinite(x@x).all(); print(torch.__version__,transformers.__version__)'),
        'phase-judge': ('import torch,vllm,transformers; assert torch.cuda.is_available(); '
                        'print(vllm.__version__,transformers.__version__)'),
    }
    checks = []
    for name, code in commands.items():
        python = root / '.envs' / name / 'bin' / 'python'
        if not python.exists():
            continue  # The version check already reports the missing environment.
        try:
            output = subprocess.check_output([str(python), '-c', code], cwd=root,
                                             text=True, stderr=subprocess.STDOUT, timeout=120).strip()
            checks.append(check(f'runtime_{name}', 'pass', output.splitlines()[-1]))
        except Exception as error:
            detail = getattr(error, 'output', None) or str(error)
            checks.append(check(f'runtime_{name}', 'fail', str(detail)[-2000:]))
    return checks


def git_output(*args):
    return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.resolve().as_posix()}',
                                    '-C', str(ROOT), *args], text=True).strip()


def online_access(models):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    identity = api.whoami()
    accessed = {}
    for model in models:
        info = api.model_info(model)
        # Fetching config.json verifies gated-file access without downloading weights.
        hf_hub_download(model, 'config.json', revision=info.sha)
        accessed[model] = info.sha
    accessed['THUDM/LongBench-v2'] = api.dataset_info('THUDM/LongBench-v2').sha
    accessed['Salesforce/wikitext'] = api.dataset_info('Salesforce/wikitext').sha
    return {'account': identity.get('name') or identity.get('fullname') or 'authenticated',
            'revisions': accessed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, default=5)
    parser.add_argument('--minimum-gpu-memory-gib', type=float, default=130)
    parser.add_argument('--minimum-free-disk-gib', type=float, default=350)
    parser.add_argument('--recommended-host-ram-gib', type=float, default=512)
    parser.add_argument('--online', action='store_true', help='verify HF login and gated config access')
    parser.add_argument('--out', default='outputs/h200-preflight.json')
    args = parser.parse_args()
    if min(args.gpus, args.minimum_gpu_memory_gib, args.minimum_free_disk_gib,
           args.recommended_host_ram_gib) <= 0:
        parser.error('GPU count and capacity thresholds must be positive')

    checks = []
    checks.append(check('platform', 'pass' if sys.platform.startswith('linux') else 'fail', sys.platform))
    try:
        commit = git_output('rev-parse', 'HEAD')
        dirty = git_output('status', '--porcelain', '--untracked-files=no')
        checks.append(check('git_tracked_files', 'fail' if dirty else 'pass', dirty or commit))
    except Exception as error:
        commit = None
        checks.append(check('git_tracked_files', 'fail', error))

    disk = shutil.disk_usage(ROOT)
    free_disk_gib = disk.free / 2**30
    ram_gib = (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
               if hasattr(os, 'sysconf') else 0.)
    checks.extend(evaluate_host_capacity(free_disk_gib, ram_gib, args.minimum_free_disk_gib,
                                         args.recommended_host_ram_gib))
    environments, env_checks = environment_versions(ROOT)
    checks.extend(env_checks)

    devices, runtime = [], {}
    try:
        import torch
        runtime = {'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                   'cudnn': torch.backends.cudnn.version(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        if not torch.cuda.is_available():
            raise RuntimeError('torch.cuda.is_available() is false')
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            # A real BF16 GEMM catches broken driver/runtime combinations that inventory alone misses.
            x = torch.randn((256, 256), device=f'cuda:{i}', dtype=torch.bfloat16)
            y = x @ x
            torch.cuda.synchronize(i)
            if not torch.isfinite(y).all().item():
                raise RuntimeError(f'GPU {i} produced nonfinite BF16 preflight output')
            devices.append({'index': i, 'name': p.name, 'total_memory_gib': p.total_memory / 2**30,
                            'compute_capability': [p.major, p.minor],
                            'bf16_supported': bool(torch.cuda.is_bf16_supported())})
        checks.extend(evaluate_gpu_inventory(devices, args.gpus, args.minimum_gpu_memory_gib))
        selected = min(args.gpus, len(devices))
        missing_peer = [[i, j] for i in range(selected) for j in range(selected) if i != j
                        and not torch.cuda.can_device_access_peer(i, j)]
        checks.append(check('gpu_peer_access', 'warn' if missing_peer else 'pass',
                            f'unavailable pairs: {missing_peer}' if missing_peer else 'all selected GPU pairs have peer access'))
        checks.append(check('bf16_gemm', 'pass', f'completed on {len(devices)} visible GPUs'))
    except Exception as error:
        checks.append(check('cuda_runtime', 'fail', error))

    checks.extend(environment_runtime_checks(ROOT))
    for name, command in (('nvcc', ['nvcc', '--version']),
                          ('nvidia_smi', ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'])):
        try:
            output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
            checks.append(check(name, 'pass', output[-2000:]))
        except Exception as error:
            checks.append(check(name, 'fail', error))

    try:
        subprocess.run([str(ROOT / '.envs/phase-baselines/bin/python'), '-m', 'scripts.fetch_baselines',
                        '--methods', 'freekv', 'factory', 'rocketkv', '--verify-only'],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        checks.append(check('upstream_revisions', 'pass', 'FreeKV, KVCache-Factory, and RocketKV match lock file'))
    except Exception as error:
        checks.append(check('upstream_revisions', 'fail', error))

    access = None
    if args.online:
        try:
            defaults = json.loads((ROOT / 'configs/phase_one.json').read_text())
            access = online_access([*defaults['models'], defaults['judge_model']])
            checks.append(check('huggingface_access', 'pass',
                                f"authenticated as {access['account']}; gated configs readable"))
        except Exception as error:
            checks.append(check('huggingface_access', 'fail', error))
    else:
        checks.append(check('huggingface_access', 'warn', 'not checked; rerun with --online after login'))

    report = {'schema': 'pagedkv-h200-preflight-v1',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'commit': commit,
              'ready': not any(c['status'] == 'fail' for c in checks),
              'checks': checks, 'runtime': runtime, 'gpus': devices,
              'host': {'free_disk_gib': free_disk_gib, 'ram_gib': ram_gib},
              'environments': environments, 'online_access': access}
    atomic_json(args.out, report)
    for item in checks:
        print(f"[{item['status'].upper():4}] {item['name']}: {item['detail']}")
    print(f"Report: {Path(args.out).resolve()}")
    if not report['ready']:
        raise SystemExit('H200 preflight failed; resolve FAIL checks before smoke runs')


if __name__ == '__main__':
    main()
