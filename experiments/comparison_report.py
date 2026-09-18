"""Collect completed/partial/missing comparisons without pooling incompatible runs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from .benchmark_state import ROOT, atomic_json, digest


def collect(plan):
    quality, system, status = [], [], []
    for job in plan['jobs']:
        path = Path(job['output'])
        state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        job_status = state.get('status', job['status'])
        if 'completed' in state:
            job_status = 'completed' if state['completed'] else 'partial'
        # A worker can fail before updating an older result (e.g. identity mismatch).
        if job.get('returncode', 0) != 0:
            job_status = 'failed'
        if state and state.get('manifest_sha256') != job['manifest_sha256']:
            raise ValueError('report/plan manifest mismatch')
        reason = state.get('error', job.get('error', job.get('reason', '')))
        status.append(dict(job=job['name'], method=job['method'], kind=job['kind'], status=job_status,
                           reason=reason, report=str(path)))
        base = dict(job=job['name'], model=state.get('model', state.get('config', {}).get('model', '')),
                    method=job['method'], status=job_status, manifest_sha256=job['manifest_sha256'],
                    configuration_sha256=digest(state.get('config', {})),
                    protocol=state.get('protocol', state.get('dataset_protocol', '')),
                    gpu_count=job['layout']['gpus'])
        complete = job_status == 'completed'
        if state.get('schema') == 'h200-benchmark-v1':
            if complete and {r['id'] for r in state['rows']} != set(state['expected']):
                raise ValueError('completed report is missing work')
            if job['kind'] == 'system' and state['rows']:
                row = dict(**base, context=state['config']['context'], batch=state['config']['batch_size'],
                           samples=len(state['rows']), workload=state.get('workload'), bandwidth_utilization_pct=None)
                for key in ('ttft_s', 'decode_ms_per_token', 'decode_tokens_per_s', 'end_to_end_tokens_per_s',
                            'peak_gpu_allocated_gb', 'peak_gpu_reserved_gb'):
                    values = [r[key] for r in state['rows'] if r.get(key) is not None]
                    row[key] = statistics.median(values) if complete and values else None
                system.append(row)
            elif job['kind'] == 'quality':
                groups = defaultdict(list)
                for row in state['rows']:
                    groups[(row['style'], row.get('metric', 'qa_f1'))].append(row)
                for (task, metric), rows in groups.items():
                    quality.append(dict(**base, arm=job['method'], task=task, metric=metric, n=len(rows),
                        score_pct=100 * statistics.mean(r.get('score', r['token_f1']) for r in rows) if complete else None))
        elif 'answers' in state:
            if complete and len(state['completed_documents']) != state['expected_documents']:
                raise ValueError('legacy report claims completion with missing documents')
            groups = defaultdict(list)
            for row in state['answers']:
                groups[(row['style'], row['arm'], row.get('metric', 'qa_f1'))].append(row)
            for (task, arm, metric), rows in groups.items():
                quality.append(dict(**base, arm=arm, task=task, metric=metric, n=len(rows),
                    score_pct=100 * statistics.mean(r.get('score', r['token_f1']) for r in rows) if complete else None))
    return dict(quality=quality, system=system, status=status)


def markdown_table(rows, fields):
    def cell(value):
        if value is None:
            return 'N/A'
        if isinstance(value, float):
            return f'{value:.3f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join('---' for _ in fields) + ' |'] +
                     ['| ' + ' | '.join(cell(row.get(f)) for f in fields) + ' |' for row in rows])


def write_reports(plan, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = collect(plan)
    atomic_json(out / 'comparison.json', result)
    for name, rows in result.items():
        fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
        with (out / f'{name}.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(rows)
    text = '# H200 comparison\n\nOnly complete jobs receive aggregate scores. N/A is unmeasured, unsupported or incomplete.\n\n'
    text += '## Quality\n\n' + markdown_table(result['quality'],
        ['model', 'method', 'arm', 'task', 'metric', 'score_pct', 'n', 'status', 'job'])
    text += '\n\n## Isolated system measurements\n\n' + markdown_table(result['system'],
        ['model', 'method', 'gpu_count', 'context', 'batch', 'decode_ms_per_token', 'ttft_s',
         'decode_tokens_per_s', 'peak_gpu_allocated_gb', 'bandwidth_utilization_pct', 'status'])
    text += '\n\nMemory is the sum of per-device PyTorch allocator peaks, not whole-device NVML memory. '
    text += 'The workload repeats one prompt across the batch. Ours quality telemetry is excluded from this table. '
    text += 'FlashAttention-2/SDPA are named in configurations; no FlashAttention-3 result is claimed.\n'
    text += '\n## Coverage and resume status\n\n' + markdown_table(result['status'], ['job', 'status', 'reason'])
    registry = json.loads((ROOT / 'configs/baseline_registry.json').read_text())
    text += '\n\n## Requested baseline registry\n\n' + markdown_table(
        [dict(method=name, **entry) for name, entry in registry.items()], ['method', 'category', 'status'])
    text += '\n\n## Interpretation\n\n'
    text += ('Compare matching model revisions, manifests, precision, budgets, GPU counts and prompt protocols. '
             'The suite selects FP16 across methods. The local runner uses question-chunk processing; official bridges use tokenwise questions, '
             'so these rows must not be treated as a fully controlled paired comparison yet. '
             'No global LongBench average is emitted from missing tasks or mixed metrics. '
             'RULER effective-context length, symbolic math grading, PPL, FA3, and the remaining official adapters '
             'require their named protocols; no metric is inferred from QA F1.\n')
    (out / 'comparison.md').write_text(text, encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    write_reports(json.loads(Path(args.plan).read_text()), args.out_dir)


if __name__ == '__main__':
    main()
