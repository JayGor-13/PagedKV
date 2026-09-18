"""Prepare, run/resume, and report the four-model FreeKV benchmark comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from .benchmark_state import ROOT, atomic_json, digest
from .freekv_protocol import FREEKV, format_prompt, chat_ids
from .gpu_layout import choose_layout
from .phase_one_backends import LABELS, support_reason

DEFAULTS = json.loads((ROOT / 'configs/phase_one.json').read_text())


def prepare(out, models, smoke=False):
    from scripts.fetch_baselines import verify
    upstream = verify('freekv')
    folder = Path(out) / 'manifests'
    folder.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi
    api = HfApi()
    lock_path = Path(out) / 'revisions.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if set(lock['models']) != set(models) or lock['smoke'] != smoke:
            raise ValueError('model selection or smoke mode changed; use another output directory')
    else:
        lock = dict(models={m: api.model_info(m).sha for m in models}, smoke=smoke,
                    longbenchv2=api.dataset_info('THUDM/LongBench-v2').sha,
                    calibration=api.dataset_info('Salesforce/wikitext').sha,
                    judge_model=DEFAULTS['judge_model'], judge_revision=api.model_info(DEFAULTS['judge_model']).sha)
        atomic_json(lock_path, lock)
    paths = [folder / f'{m.split("/")[-1]}-{bench}.json' for m in models for bench in DEFAULTS['benchmarks']]
    if all(p.exists() for p in paths):
        for p in paths:
            data = json.loads(p.read_text())
            if data['upstream'] != upstream or data['revision'] != lock['models'][data['model']]:
                raise ValueError('existing manifest provenance changed')
        return paths
    from datasets import load_dataset
    from transformers import AutoTokenizer
    lb = list(load_dataset('THUDM/LongBench-v2', revision=lock['longbenchv2'], split='train'))
    lg_path = FREEKV / 'accuracy/eval/LongGenBench/Dataset_short.json'
    lg = json.loads(lg_path.read_text(encoding='utf-8'))
    # Separate training corpus for codec calibration, never evaluation answers.
    calibration = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', revision=lock['calibration'], split='train')
    calibration_text = '\n'.join(row['text'] for row in calibration.select(range(min(1000, len(calibration)))))
    specs = {m['id']: m for m in json.loads((ROOT / 'configs/h200_models.json').read_text())['models']}
    for model in models:
        tokenizer = AutoTokenizer.from_pretrained(model, revision=lock['models'][model], use_fast=False)
        calib_ids = tokenizer.encode(calibration_text)[:4096]
        if len(calib_ids) < 4096:
            raise ValueError('not enough calibration tokens')
        for bench, raw in [('longbenchv2', lb), ('longgenbench', lg)]:
            path = folder / f'{model.split("/")[-1]}-{bench}.json'
            if path.exists():
                continue
            examples = []
            for index, item in enumerate(raw[:2] if smoke else raw):
                prompt = format_prompt(item, bench)
                ids, stats = chat_ids(tokenizer, prompt, model, bench, cap=4096 if smoke else 120000)
                data = {k: v for k, v in item.items() if k not in ('context', 'prompt')}
                examples.append(dict(id=str(item.get('_id', index)), token_ids=ids, data=data, **stats))
            manifest = dict(schema='freekv-protocol-v1', model=model, revision=lock['models'][model],
                            benchmark=bench, smoke=smoke, seed=42, upstream=upstream, examples=examples,
                            source_count=len(raw), evaluated_count=len(examples),
                            source_revision=lock['longbenchv2'] if bench == 'longbenchv2' else upstream['commit'],
                            calibration=[calib_ids[i:i+1024] for i in range(0, 4096, 1024)],
                            calibration_source=dict(dataset='Salesforce/wikitext', split='train', revision=lock['calibration']),
                            stop_ids=tokenizer('*** finished', truncation=False).input_ids if bench == 'longgenbench' else [],
                            protocol=dict(name='FreeKV LongBench2 zero-shot / LongGenBench short',
                                prompt_cap_before_chat=4096 if smoke else 120000,
                                chat='FreeKV llama user / Qwen system+user', rope_extension='none',
                                released_context=specs[model]['context_tokens'],
                                extrapolation_policy='FreeKV 120K protocol; Qwen >32K is explicit unscaled extrapolation',
                                sampling='CPU multinomial; per-example SHA256-derived seed, independent of resume order',
                                eos='stop also on first EOS (upstream only checks after the second token)',
                                longgen_stop='upstream tokenizer(stop_sign) tokenization, including default special tokens'))
            atomic_json(path, manifest)
            print(f'Prepared {path}: {len(examples)} examples', flush=True)
    return paths


def make_plan(paths, out, gpus, memory, methods, env_root):
    specs = {m['id']: m for m in json.loads((ROOT / 'configs/h200_models.json').read_text())['models']}
    if not methods or set(methods) - set(LABELS) or len(set(methods)) != len(methods):
        raise ValueError('unknown or duplicate method')
    jobs = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        spec = specs[data['model']]
        length = max(len(e['token_ids']) for e in data['examples']) + (8 if data['smoke'] else 16000 if data['benchmark'] == 'longgenbench' else 128)
        for method in methods:
            layout = choose_layout('ours' if method == 'ours' else 'freekv', spec, gpus, memory, length)
            reason = support_reason(method, spec['architecture'], layout['gpus'])
            if layout['status'] != 'ready':
                reason = layout['reason']
            name = f'{data["model"].split("/")[-1]}/{data["benchmark"]}/{method}'
            output = Path(out).resolve() / 'results' / name / 'result.json'
            python = Path(env_root).resolve() / ('phase-ours' if method == 'ours' else 'phase-baselines') / 'bin/python'
            argv = [str(python), '-m', 'experiments.phase_one_worker', '--manifest', str(Path(path).resolve()),
                    '--method', method, '--out', str(output), '--gpus', str(layout['gpus']),
                    '--gpu-memory-gib', str(int(memory * .9)), '--resume']
            jobs.append(dict(name=name, model=data['model'], benchmark=data['benchmark'], method=method,
                             implementation=LABELS[method], manifest=str(Path(path).resolve()), manifest_sha256=digest(data),
                             output=str(output), layout=layout, status='unsupported' if reason else 'planned',
                             reason=reason, argv=[] if reason else argv))
    return dict(schema='phase-one-plan-v1', gpus=gpus, memory_gib=memory, jobs=jobs)


def write_report(plan, out):
    rows = []
    for job in plan['jobs']:
        path = Path(job['output'])
        result = json.loads(path.read_text()) if path.exists() else {}
        samples = result.get('rows', [])
        complete = (job['status'] != 'failed' and result.get('status') == 'completed'
                    and len(samples) == len(result.get('expected', [])) and bool(samples)
                    and result.get('manifest_sha256', job['manifest_sha256']) == job['manifest_sha256'])
        row = dict(model=job['model'], benchmark=job['benchmark'], method=job['implementation'],
                   status='failed' if job['status'] == 'failed' else result.get('status', job['status']), samples=len(samples),
                   expected=len(result.get('expected', [])), smoke=result.get('smoke'),
                   accuracy=None, completion_rate=None, accuracy_once=None, accuracy_range=None,
                   accuracy_periodic=None, average_accuracy=None,
                   prompt_archive_samples=sum(r.get('prompt_archive_used', False) for r in samples) if job['method'] == 'ours' else None,
                   reason=job.get('reason') or result.get('error'))
        if complete and not result.get('smoke') and job['benchmark'] == 'longbenchv2':
            row['accuracy'] = 100 * sum(r['accuracy'] for r in samples) / len(samples)
        if complete and not result.get('smoke') and job['benchmark'] == 'longgenbench':
            row['completion_rate'] = sum(r['completion_rate'] for r in samples) / len(samples)
            judge_path = path.with_name('judge.json')
            judge = json.loads(judge_path.read_text()) if judge_path.exists() else {}
            if judge.get('status') == 'completed' and judge.get('generation_sha256') == digest(result):
                for category in ('once', 'range', 'periodic'):
                    answers = [r['answer'] for r in judge['rows'] if r.get('category') == category]
                    row[f'accuracy_{category}'] = sum(a == 'yes' for a in answers) / len(answers) if answers else 0.
                row['average_accuracy'] = sum(row[f'accuracy_{c}'] for c in ('once', 'range', 'periodic')) / 3
            else:
                row['reason'] = 'Generation complete; LongGenBench judge pending'
        rows.append(row)
    output = Path(out)
    atomic_json(output / 'comparison.json', rows)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['status'])
        writer.writeheader()
        writer.writerows(rows)
    def value(v):
        return '—' if v is None else f'{v:.4f}' if isinstance(v, float) else str(v).replace('|', '/')
    columns = ['model', 'benchmark', 'method', 'status', 'samples', 'accuracy', 'completion_rate', 'average_accuracy', 'reason']
    text = 'Quality results only. Accuracy/completion are percentages; LongGenBench judged accuracy is 0–1. Smoke results are not benchmark scores.\n\n'
    text += '| ' + ' | '.join(columns) + ' |\n|' + '|'.join(['---']*len(columns)) + '|\n'
    text += '\n'.join('| ' + ' | '.join(value(r[c]) for c in columns) + ' |' for r in rows)
    (output / 'comparison.md').write_text(text+'\n', encoding='utf-8')


def execute(plan, out):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else [str(i) for i in range(plan['gpus'])]
    if len(devices) < plan['gpus']:
        raise ValueError('--gpus exceeds CUDA_VISIBLE_DEVICES')
    pending = [j for j in plan['jobs'] if j['argv']]
    def one(job, assigned):
        path = Path(job['output'])
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(Path(job['manifest']).read_text())
        if digest(data) != job['manifest_sha256']:
            raise ValueError('frozen manifest changed since planning')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(assigned), TOKENIZERS_PARALLELISM='false')
        with path.with_suffix('.log').open('a', encoding='utf-8') as log:
            return subprocess.run(job['argv'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    while pending:
        wave, used = [], 0
        while pending and used + pending[0]['layout']['gpus'] <= plan['gpus']:
            if pending[0]['method'] == 'ours' and any(j['method'] == 'ours' for j, _ in wave):
                break  # the diagnostic archive has substantial host-RAM demand
            job = pending.pop(0)
            n = job['layout']['gpus']
            wave.append((job, devices[used:used+n]))
            used += n
        if not wave:
            raise ValueError('job requires more GPUs than available')
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            futures = [(j, pool.submit(one, j, ids)) for j, ids in wave]
            for job, future in futures:
                try:
                    job['returncode'] = future.result()
                    result_path = Path(job['output'])
                    result = json.loads(result_path.read_text()) if result_path.exists() else {}
                    valid = result.get('status') == 'completed' and bool(result.get('expected')) and len(result.get('rows', [])) == len(result['expected'])
                    job['status'] = 'completed' if job['returncode'] == 0 and valid else 'failed'
                except Exception as error:
                    job.update(status='failed', reason=str(error))
                atomic_json(Path(out) / 'plan.json', plan)
                write_report(plan, out)
                print(f'{job["name"]}: {job["status"]}', flush=True)
    return not any(j['status'] == 'failed' for j in plan['jobs'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'run', 'report'))
    p.add_argument('--out', default='outputs/phase-one')
    p.add_argument('--models', nargs='+', default=DEFAULTS['models'])
    p.add_argument('--methods', nargs='+', default=DEFAULTS['methods'])
    p.add_argument('--gpus', type=int, default=5)
    p.add_argument('--gpu-memory-gib', type=int, help='Default: detect the smallest selected GPU; plan-only uses 140 GiB')
    p.add_argument('--env-root', default='.envs')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    if args.gpus < 1:
        p.error('--gpus must be positive')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Advisory lock is automatically released on process death, including SIGKILL.
    # A stale lock filename does not prevent resumption.
    import fcntl
    with (out / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another process is using this output directory')
        if args.action == 'report':
            write_report(json.loads((out / 'plan.json').read_text()), out)
            return
        memory = args.gpu_memory_gib or DEFAULTS['gpu_memory_gib']
        if args.action == 'run' and not args.plan_only:
            import torch
            if torch.cuda.device_count() < args.gpus:
                raise ValueError('fewer visible CUDA devices than --gpus')
            detected = int(min(torch.cuda.get_device_properties(i).total_memory for i in range(args.gpus)) / 2**30)
            memory = args.gpu_memory_gib or detected
            if memory > detected or memory <= 0:
                raise ValueError(f'--gpu-memory-gib must be in 1..{detected} for the selected GPUs')
            print(f'Using {args.gpus} GPUs; planning with {memory} GiB each (10% reserve).', flush=True)
        paths = prepare(out, args.models, args.smoke)
        if args.action == 'prepare':
            return
        plan = make_plan(paths, out, args.gpus, memory, args.methods, args.env_root)
        plan_path = out / 'plan.json'
        if plan_path.exists():
            previous = json.loads(plan_path.read_text())
            before = [(j['name'], j['manifest_sha256'], j['argv']) for j in previous['jobs']]
            after = [(j['name'], j['manifest_sha256'], j['argv']) for j in plan['jobs']]
            if before != after:
                if any(Path(j['output']).exists() for j in previous['jobs']):
                    raise ValueError('plan changed (models, methods, paths or GPU layout); use a new output directory')
            else:
                plan = previous
                plan['gpus'] = args.gpus  # scheduling concurrency may change if per-job layout stays identical
        atomic_json(plan_path, plan)
        write_report(plan, out)
        print(f'{len(plan["jobs"])} jobs, {sum(bool(j["argv"]) for j in plan["jobs"])} executable. See {plan_path}')
        if not args.plan_only and not execute(plan, out):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
