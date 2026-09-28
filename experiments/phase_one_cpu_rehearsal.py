"""Run the phase-one planner, scheduler, checkpoint/resume, and reporter on CPU.

This is an orchestration test. It never loads a model and never validates CUDA,
native baseline kernels, numerical equivalence, model quality, or performance.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from .benchmark_state import append_row, atomic_json
from .cpu_rehearsal_worker import open_worker_checkpoint, rehearsal_row
from .phase_one import DEFAULTS, execute, make_plan, require_complete_plan, write_report


def write_manifests(out, models):
    folder = Path(out) / 'manifests'
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for model in models:
        for benchmark in DEFAULTS['benchmarks']:
            path = folder / f'{model.split("/")[-1]}-{benchmark}.json'
            manifest = dict(schema='cpu-orchestration-rehearsal-v1', model=model,
                            revision='0' * 40, benchmark=benchmark, smoke=True,
                            examples=[dict(id=f'{benchmark}-0', token_ids=list(range(32))),
                                      dict(id=f'{benchmark}-1', token_ids=list(range(48)))],
                            source_count=2, evaluated_count=2,
                            warning='Synthetic manifest for CPU orchestration rehearsal only')
            atomic_json(path, manifest)
            paths.append(path)
    return paths


def rehearse(out, models=None, methods=None, gpus=5, memory_gib=140):
    out = Path(out).resolve()
    models = list(models or DEFAULTS['models'])
    methods = list(methods or DEFAULTS['methods'])
    paths = write_manifests(out, models)
    plan = make_plan(paths, out, gpus, memory_gib, methods, out / 'unused-envs')
    require_complete_plan(plan, models, methods)
    for job in plan['jobs']:
        job['argv'] = [sys.executable, '-m', 'experiments.cpu_rehearsal_worker',
                       '--manifest', job['manifest'], '--method', job['method'],
                       '--out', job['output'], '--resume']
        job['status'] = 'planned'
        job['reason'] = None
    atomic_json(out / 'plan.json', plan)
    write_report(plan, out)

    # Seed one committed row to model a process that stopped between examples.
    first = plan['jobs'][0]
    first_output = Path(first['output'])
    if not first_output.exists():
        manifest, state = open_worker_checkpoint(first['manifest'], first['method'],
                                                 first_output, resume=False)
        append = rehearsal_row(manifest['examples'][0], manifest['benchmark'], first['method'])
        append_row(first_output, state, append)

    previous_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(gpus)))
    try:
        if not execute(plan, out):
            raise RuntimeError('first CPU scheduler pass failed; inspect result.log files')
        if not execute(plan, out):
            raise RuntimeError('completed-checkpoint resume pass failed')
    finally:
        if previous_visible is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = previous_visible
    write_report(plan, out)
    rows = json.loads((out / 'comparison.json').read_text(encoding='utf-8'))
    expected = len(models) * len(DEFAULTS['benchmarks']) * len(methods)
    if len(rows) != expected or any(row['status'] != 'completed' for row in rows):
        raise RuntimeError('CPU rehearsal report is incomplete')
    if any(row['accuracy'] is not None or row['completion_rate'] is not None for row in rows):
        raise RuntimeError('CPU rehearsal must not publish benchmark scores')
    summary = dict(schema='phase-one-cpu-rehearsal-v1', completed=True,
                   jobs=expected, scheduler_passes=2, seeded_partial_checkpoint=True,
                   validates=['matrix planning', 'GPU-slot scheduling', 'atomic checkpoints',
                              'partial resume', 'completed-job resume', 'combined reports'],
                   does_not_validate=['model loading', 'CUDA', 'FlashAttention',
                                      'native baseline kernels', 'multi-GPU dispatch',
                                      'accuracy', 'latency'])
    atomic_json(out / 'rehearsal-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='outputs/cpu-phase-one-rehearsal')
    parser.add_argument('--models', nargs='+', default=DEFAULTS['models'])
    parser.add_argument('--methods', nargs='+', default=DEFAULTS['methods'])
    parser.add_argument('--gpus', type=int, default=5,
                        help='simulated scheduler slots; no GPU is accessed')
    parser.add_argument('--gpu-memory-gib', type=int, default=140,
                        help='simulated per-GPU capacity passed to the planner')
    args = parser.parse_args()
    if args.gpus < 1 or args.gpu_memory_gib < 1:
        parser.error('simulated GPU count and memory must be positive')
    summary = rehearse(args.out, args.models, args.methods, args.gpus,
                       args.gpu_memory_gib)
    print(f'CPU orchestration rehearsal completed: {summary["jobs"]} jobs, two scheduler passes.')
    print('No model, CUDA kernel, accuracy, or performance result was produced.')
    print(f'Report: {Path(args.out).resolve() / "rehearsal-summary.json"}')


if __name__ == '__main__':
    main()
