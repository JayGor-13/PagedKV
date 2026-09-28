"""Dependency-free worker used only to rehearse orchestration on a CPU host.

It deliberately produces no model text or benchmark score. Its result schema and
atomic checkpoint behavior match the phase-one launcher closely enough to test
planning, scheduling, interruption recovery, and report generation.
"""
import argparse
import json
import platform
from pathlib import Path

from .benchmark_state import append_row, digest, open_checkpoint, source_identity


def worker_identity(manifest, method):
    return dict(kind='cpu-orchestration-rehearsal', manifest_sha256=digest(manifest),
                method=method, python=platform.python_version(), source=source_identity())


def rehearsal_row(example, benchmark, method):
    row = dict(id=example['id'], rehearsal=True, generated_tokens=0,
               prompt_archive_used=method == 'ours')
    if benchmark == 'longbenchv2':
        row['accuracy'] = 0
    else:
        row['completion_rate'] = 0
    return row


def open_worker_checkpoint(manifest_path, method, output, resume):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    expected = [row['id'] for row in manifest['examples']]
    metadata = dict(manifest_sha256=digest(manifest), model=manifest['model'],
                    benchmark=manifest['benchmark'], method=method, smoke=True,
                    rehearsal=True,
                    warning='CPU orchestration rehearsal; no model, CUDA kernel, or benchmark score')
    state = open_checkpoint(output, worker_identity(manifest, method), expected,
                            metadata, resume=resume)
    return manifest, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    manifest, state = open_worker_checkpoint(args.manifest, args.method, args.out, args.resume)
    completed = {row['id'] for row in state['rows']}
    for example in manifest['examples']:
        if example['id'] not in completed:
            append_row(args.out, state,
                       rehearsal_row(example, manifest['benchmark'], args.method))
    print(f'{manifest["model"]}/{manifest["benchmark"]}/{args.method}: '
          f'{len(state["rows"])}/{len(state["expected"])} CPU rehearsal rows')


if __name__ == '__main__':
    main()
