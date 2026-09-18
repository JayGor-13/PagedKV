"""Create/run a resumable, GPU-count-aware comparison plan across isolated Pythons."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from pathlib import Path
import subprocess
import sys

from .benchmark_state import ROOT, atomic_json, digest
from .gpu_layout import choose_layout


def make_plan(manifests, methods, environments, gpus, memory_gib, out_dir,
              kinds=('quality', 'system'), context=65536, batch=8, ablations=False):
    registry = json.loads((ROOT / 'configs/baseline_registry.json').read_text())
    specs = {m['id']: m for m in json.loads((ROOT / 'configs/h200_models.json').read_text())['models']}
    if len(methods) != len(set(methods)):
        raise ValueError('duplicate methods')
    jobs = []
    for manifest in manifests:
        path = Path(manifest).resolve()
        data = json.loads(path.read_text(encoding='utf-8'))
        if not re.fullmatch(r'[0-9a-fA-F]{40}', data.get('revision') or ''):
            raise ValueError('pin model revision in each frozen manifest')
        if data['model'] not in specs:
            raise ValueError('add model architecture and size to configs/h200_models.json before planning')
        spec = specs[data['model']]
        from .upstream_runner import quality_cases
        cases = quality_cases(data)
        quality_length = max(len(d['token_ids']) + len(q['token_ids']) + q.get('max_new_tokens', 128) for _, d, q in cases)
        for method in methods:
            if method not in registry:
                raise ValueError(f'unknown method: {method}')
            for kind in kinds:
                variants = [('base', [])]
                if ablations and method == 'ours' and kind == 'quality':
                    variants = [(f'page{p}', ['--page', str(p), '--recall-k', str(1024 // p)]) for p in (16, 32, 64, 128)]
                for variant, extra in variants:
                    tag = f'{data["model"].split("/")[-1]}/{path.stem}-{digest(data)[:10]}/{method}/{kind}/{variant}'
                    output = Path(out_dir).resolve() / tag / 'result.json'
                    length = quality_length if kind == 'quality' else context + 128
                    layout = choose_layout(method, spec, gpus, memory_gib, length, 1 if kind == 'quality' else batch)
                    job = dict(name=tag, method=method, kind=kind, manifest=str(path),
                               manifest_sha256=digest(data), output=str(output), layout=layout,
                               status='planned', argv=[])
                    reason = None
                    if method not in ('ours', 'full', 'arkvale', 'kivi'):
                        reason = registry[method]['status']
                    elif method == 'ours' and kind == 'system':
                        reason = 'current ours runner retains diagnostic caches; isolated serving implementation pending'
                    elif method == 'arkvale':
                        reason = 'unmodified ArkVale lacks Llama-3.1 RoPE and Qwen2 architecture support'
                    elif method == 'kivi' and spec['architecture'] != 'llama':
                        reason = 'official KIVI has no Qwen2 implementation'
                    elif length > spec['context_tokens']:
                        reason = f'{length} total tokens exceeds validated context {spec["context_tokens"]}'
                    elif layout['status'] != 'ready':
                        reason = layout['reason']
                    elif method not in environments:
                        reason = f'no Python executable configured for {method}'
                    if reason:
                        job.update(status='unsupported', reason=reason)
                    else:
                        argv = ['-m', 'experiments.run_rare_facts' if method == 'ours' else 'experiments.upstream_runner',
                                '--model', data['model'], '--revision', data['revision'], '--manifest', str(path),
                                '--out', str(output), '--resume', '--gpus', str(layout['gpus']),
                                '--gpu-memory-gib', str(int(memory_gib * .9))]
                        if method == 'ours':
                            argv += ['--device', 'cuda', '--dtype', 'float16', '--selection', 'random,scan', '--band', 'all', '--stop-eos',
                                     '--calibration-dir', str(Path(out_dir).resolve() / 'calibration'),
                                     '--svd-method', 'randomized', '--rank-cap', '1024', '--topk', '256',
                                     '--hot-tokens', '1024', '--halo', '0', *extra]
                        else:
                            argv += ['--method', method, '--kind', kind, '--context', str(context), '--batch-size', str(batch)]
                        job['argv'] = [environments[method], *argv]
                    jobs.append(job)
    names = [j['name'] for j in jobs]
    if len(set(names)) != len(names):
        raise ValueError('duplicate manifests/job output collision')
    return dict(schema='h200-suite-v1', gpus=gpus, memory_gib=memory_gib, jobs=jobs,
                notes=['Single-GPU jobs can run concurrently on distinct GPUs; sharded jobs reserve their full GPU count.',
                       'Memory estimates are screening estimates, not a fit guarantee. Ours uses randomized rank-1024 calibration.',
                       'No weight quantization, CPU model offload, RoPE extension, or substitute baselines are enabled automatically.'])


def execute(plan, plan_path):
    """Waves avoid GPU sharing. Failed jobs do not suppress remaining comparisons."""
    jobs = [j for j in plan['jobs'] if j['argv']]
    def one(job, devices):
        output = Path(job['output'])
        output.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
        # Native workers and the local quality runner validate resume identities.
        with output.with_suffix('.log').open('a', encoding='utf-8') as log:
            return subprocess.run(job['argv'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else [str(i) for i in range(plan['gpus'])]
    if len(devices) < plan['gpus']:
        raise ValueError('--gpus exceeds CUDA_VISIBLE_DEVICES')
    while jobs:
        wave, used = [], 0
        while jobs and used + jobs[0]['layout']['gpus'] <= plan['gpus']:
            if wave and (jobs[0]['kind'] == 'system' or wave[0][0]['kind'] == 'system' or
                         (jobs[0]['method'] == 'ours' and any(j['method'] == 'ours' for j, _ in wave))):
                break
            job = jobs.pop(0)
            n = job['layout']['gpus']
            wave.append((job, devices[used:used+n]))
            used += n
            if job['kind'] == 'system':
                break
        if not wave:
            raise ValueError('job requests more GPUs than plan')
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = [(job, pool.submit(one, job, assigned)) for job, assigned in wave]
            for job, future in futures:
                try:
                    code = future.result()
                    report = json.loads(Path(job['output']).read_text()) if Path(job['output']).exists() else {}
                    job['status'] = report.get('status', 'completed' if code == 0 and report.get('completed') else 'failed')
                    job['returncode'] = code
                    if code != 0 and job['status'] == 'completed':
                        job['status'] = 'failed'
                except Exception as error:
                    job.update(status='failed', error=str(error))
                atomic_json(plan_path, plan)
                print(f'{job["name"]}: {job["status"]}', flush=True)
    from .comparison_report import write_reports
    write_reports(plan, Path(plan_path).parent / 'comparison')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--manifests', nargs='+')
    p.add_argument('--methods', default='ours,full,kivi,arkvale')
    p.add_argument('--environments', help='JSON mapping method to absolute Python executable')
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--gpu-memory-gib', type=int, default=130, help='Usable device capacity; 10%% reserved in planning')
    p.add_argument('--out-dir', default='outputs/h200')
    p.add_argument('--kind', choices=('quality', 'system', 'both'), default='both')
    p.add_argument('--context', type=int, default=65536)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--ablations', action='store_true')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.gpus < 1:
        p.error('--gpus must be positive')
    path = Path(args.plan)
    if args.manifests:
        if path.exists():
            raise FileExistsError('plan already exists; resume without --manifests or use a new plan path')
        envs = json.loads(Path(args.environments).read_text()) if args.environments else {'ours': sys.executable, 'full': sys.executable}
        plan = make_plan(args.manifests, args.methods.split(','), envs, args.gpus, args.gpu_memory_gib,
                         args.out_dir, ('quality', 'system') if args.kind == 'both' else (args.kind,),
                         args.context, args.batch_size, args.ablations)
        atomic_json(path, plan)
    else:
        plan = json.loads(path.read_text())
    print(f'{len(plan["jobs"])} jobs; {sum(bool(j["argv"]) for j in plan["jobs"])} executable. Plan: {path}')
    if args.execute:
        execute(plan, path)
        if any(j['status'] == 'failed' for j in plan['jobs']):
            raise SystemExit(1)
    else:
        from .comparison_report import write_reports
        write_reports(plan, path.parent / 'comparison')


if __name__ == '__main__':
    main()
