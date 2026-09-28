"""Dependency-free atomic checkpoints and content identities for GPU jobs."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    # Windows scanners/indexers can briefly open a just-flushed file without
    # delete sharing, making an otherwise atomic replace fail with WinError 5.
    # Bound the retry so a persistent permissions problem still fails loudly.
    for attempt in range(8):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.025 * (attempt + 1))


def source_identity():
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('experiments', 'kvtc') for p in sorted((ROOT / folder).glob('*.py'))}


def open_checkpoint(path, identity, expected, metadata, resume=False):
    if len(expected) != len(set(expected)) or not expected:
        raise ValueError('expected work IDs must be unique and nonempty')
    path = Path(path)
    if path.exists():
        if not resume:
            raise FileExistsError('output exists; use --resume or a new output directory')
        state = json.loads(path.read_text(encoding='utf-8'))
        if state['run_identity'] != identity or state['expected'] != expected:
            raise ValueError('resume identity mismatch: inputs, settings, environment or source changed')
        ids = [r['id'] for r in state['rows']]
        if len(ids) != len(set(ids)) or set(ids) - set(expected):
            raise ValueError('invalid checkpoint row IDs')
        if state.get('status') == 'completed' and set(ids) != set(expected):
            raise ValueError('completed checkpoint is missing rows')
        return state
    return dict(schema='h200-benchmark-v1', run_identity=identity, expected=expected,
                status='running', rows=[], **metadata)


def append_row(path, state, row):
    if row['id'] not in state['expected'] or row['id'] in {r['id'] for r in state['rows']}:
        raise ValueError('unexpected or duplicate result')
    state['rows'].append(row)
    state['status'] = 'completed' if len(state['rows']) == len(state['expected']) else 'running'
    state.pop('error', None)
    atomic_json(path, state)
