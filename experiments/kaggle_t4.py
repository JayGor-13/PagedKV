"""Resumable, shortened small-model experiments on one Kaggle T4 (FP16/SDPA).

Separate output schema and reports: never a reproduction of full H200 results.
Each method runs in its own process/environment to isolate upstream patches.
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .benchmark_state import ROOT, atomic_json, digest, source_identity, open_checkpoint, append_row
from .freekv_protocol import FREEKV, settings, format_prompt, chat_ids, score_generation
from .phase_one_backends import LABELS, source_for

MODELS = tuple(f'Qwen/Qwen2.5-{size}-Instruct' for size in ('0.5B', '1.5B', '3B'))
PROFILE = 'kaggle-t4-shortened-v1'


def protocol(args):
    if not 1 <= args.samples <= 400 or not 512 <= args.prompt_cap <= 4096 or not 1 <= args.max_new_tokens <= 1024:
        raise ValueError('T4 limits: 1..400 samples, 512..4096 prompt tokens, 1..1024 output tokens')
    if len(set(args.models)) != len(args.models) or len(set(args.methods)) != len(args.methods):
        raise ValueError('Duplicate models/methods')
    return dict(profile=PROFILE, models=args.models, methods=args.methods,
                benchmarks=args.benchmarks, samples=1 if args.smoke else args.samples,
                prompt_cap=1024 if args.smoke else args.prompt_cap,
                max_new_tokens=8 if args.smoke else args.max_new_tokens,
                smoke=args.smoke, seed=42, dtype='float16', attention='PyTorch SDPA replacement',
                sink=64, recent=64, budget=256, page_size=32, calibration_tokens=512,
                codec_rank=64, protocol_note='First-N subset; BOTH benchmark prompts may be truncated; '
                'short generation; reduced KV budget and calibration; no H200 comparability')


def prepare(out, cfg):
    from scripts.fetch_baselines import verify
    upstream = {name: verify(name) for name in {'freekv', *(source_for(m) for m in cfg['methods'])}}
    path = out / 'inputs.json'
    if path.exists():
        frozen = json.loads(path.read_text(encoding='utf-8'))
        if frozen['config'] != cfg or frozen['upstream'] != upstream:
            raise ValueError('Configuration/upstream changed: use a new output directory')
        return frozen
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from datasets import load_dataset
    api = HfApi()
    revisions = {m: api.model_info(m).sha for m in cfg['models']}
    calibration_revision = api.dataset_info('Salesforce/wikitext').sha
    calibration = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1',
                               revision=calibration_revision, split='train')
    training_text = '\n'.join(row['text'] for row in calibration.select(range(min(1000, len(calibration)))))
    raw, sources = {}, {}
    if 'longbenchv2' in cfg['benchmarks']:
        revision = api.dataset_info('THUDM/LongBench-v2').sha
        raw['longbenchv2'] = list(load_dataset('THUDM/LongBench-v2', revision=revision, split='train'))
        sources['longbenchv2'] = revision
    if 'longgenbench' in cfg['benchmarks']:
        raw['longgenbench'] = json.loads((FREEKV / 'accuracy/eval/LongGenBench/Dataset_short.json').read_text(encoding='utf-8'))
        sources['longgenbench'] = upstream['freekv']['commit']
    manifests = []
    for model in cfg['models']:
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revisions[model], use_fast=False)
        calibration_ids = tokenizer.encode(training_text, add_special_tokens=False,
                                           truncation=True, max_length=cfg['calibration_tokens'])
        if len(calibration_ids) != cfg['calibration_tokens']:
            raise ValueError('Insufficient calibration tokens')
        for bench, rows in raw.items():
            examples = []
            for i, row in enumerate(rows[:cfg['samples']]):
                # Explicit reduced-context experiment: use head/tail truncation
                # for LongGen too; never silently claim the upstream protocol.
                ids, stats = chat_ids(tokenizer, format_prompt(row, bench), model,
                                      'longbenchv2', cap=cfg['prompt_cap'])
                if len(ids) > cfg['prompt_cap'] + 256:
                    raise ValueError('Chat template unexpectedly exceeds reserved token overhead')
                examples.append(dict(id=str(row.get('_id', i)), token_ids=ids,
                                     data={k: v for k, v in row.items() if k not in ('context', 'prompt')}, **stats))
            manifests.append(dict(schema=PROFILE, model=model, revision=revisions[model], benchmark=bench,
                                  smoke=cfg['smoke'], seed=cfg['seed'], examples=examples, source_count=len(rows),
                                  source_revision=sources[bench], calibration=[calibration_ids],
                                  calibration_revision=calibration_revision, codec_rank_cap=cfg['codec_rank'],
                                  stop_ids=tokenizer('*** finished').input_ids if bench == 'longgenbench' else []))
    frozen = dict(config=cfg, upstream=upstream, manifests=manifests)
    atomic_json(path, frozen)
    return frozen


def result_path(out, manifest, method):
    return out / 'results' / manifest['model'].split('/')[-1] / manifest['benchmark'] / method / 'result.json'


def print_failure(path):
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    print(f'FAILED: {path}', flush=True)
    if state.get('error'):
        print('Error:', state['error'], flush=True)
    log_path = path.with_suffix('.log')
    if log_path.exists():
        lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
        print('Last worker log lines:', flush=True)
        print('\n'.join(lines[-40:]), flush=True)


def write_report(out, frozen):
    failure_path = out / 'worker_failures.json'
    failures = json.loads(failure_path.read_text(encoding='utf-8')) if failure_path.exists() else {}
    records = []
    for manifest in frozen['manifests']:
        for method in frozen['config']['methods']:
            path = result_path(out, manifest, method)
            state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            rows = state.get('rows', [])
            failure = failures.get(str(path.relative_to(out)))
            valid = (not failure and state.get('status') == 'completed' and state.get('profile') == PROFILE
                     and state.get('manifest_sha256') == digest(manifest)
                     and len(rows) == len(manifest['examples'])
                     and {r['id'] for r in rows} == {e['id'] for e in manifest['examples']})
            scored = valid and not manifest['smoke']
            field = 'accuracy' if manifest['benchmark'] == 'longbenchv2' else 'completion_rate'
            records.append(dict(model=manifest['model'], benchmark=manifest['benchmark'], method=method,
                status='failed' if failure else state.get('status', 'pending') if valid or state.get('status') != 'completed' else 'invalid',
                samples=len(rows), expected=len(manifest['examples']),
                subset_accuracy_pct=100*sum(r[field] for r in rows)/len(rows) if scored and field == 'accuracy' else None,
                shortened_completion_pct=100*sum(r[field] for r in rows)/len(rows) if scored and field == 'completion_rate' else None,
                judged_accuracy=None, generated_tokens=sum(r['output_tokens'] for r in rows),
                elapsed_sample_seconds=sum(r['elapsed_seconds'] for r in rows),
                implementation=state.get('implementation', ''), error=failure or state.get('error', '')))
    atomic_json(out / 'comparison.json', dict(profile=PROFILE, config=frozen['config'], rows=records))
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    columns = ('model', 'benchmark', 'method', 'status', 'samples', 'subset_accuracy_pct', 'shortened_completion_pct')
    lines = ['# T4 shortened experiments', '', frozen['config']['protocol_note'],
             'Smoke scores are omitted. LongGenBench judged accuracy is not computed.', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join('' if r[c] is None else str(r[c]) for c in columns) + ' |' for r in records]
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def worker(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .phase_one_backends import patch_model
    from .phase_one_worker import generate
    frozen = json.loads((args.out / 'inputs.json').read_text(encoding='utf-8'))
    cfg = frozen['config']
    manifest = frozen['manifests'][args.manifest_index]
    method = args.method
    expected_version = '5.16.1' if method == 'ours' else '4.45.2'
    if transformers.__version__ != expected_version or cfg['profile'] != PROFILE or method not in cfg['methods']:
        raise ValueError('Wrong worker environment or profile/method')
    if torch.cuda.device_count() != 1:
        raise ValueError('T4 workers require exactly one visible CUDA device')
    from scripts.fetch_baselines import verify
    for name, info in frozen['upstream'].items():
        if verify(name) != info:
            raise ValueError('Upstream identity changed')
    config = dict(settings(manifest['benchmark'], method), **{k: cfg[k] for k in
                  ('sink', 'recent', 'budget', 'page_size', 'max_new_tokens')})
    packages = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'accelerate')}
    hardware = dict(name=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                    memory=torch.cuda.get_device_properties(0).total_memory)
    identity = digest([frozen, method, manifest, config, packages, hardware, source_identity()])
    path = result_path(args.out, manifest, method)
    metadata = dict(profile=PROFILE, model=manifest['model'], benchmark=manifest['benchmark'], method=method,
                    implementation=LABELS[method].replace('FlashAttention-2', 'PyTorch SDPA')
                    + (' [T4 FP16 cache/weights; FP32 retrieval attention]' if method == 'rocketkv'
                       else ' [T4 FP16 portable]'),
                    manifest_sha256=digest(manifest), smoke=manifest['smoke'], packages=packages,
                    hardware=hardware, config=config)
    state = open_checkpoint(path, identity, [e['id'] for e in manifest['examples']], metadata, resume=True)
    state['schema'] = PROFILE
    if state['status'] == 'completed':
        print('Already completed:', path, flush=True)
        return
    atomic_json(path, state)
    try:
        if method != 'ours':
            from .t4_attention import install_sdpa_bridge
            install_sdpa_bridge()
        if method == 'rocketkv':
            os.environ['PAGEDKV_ROCKET_FP32_ATTENTION'] = '1'
        tokenizer = AutoTokenizer.from_pretrained(manifest['model'], revision=manifest['revision'], use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(manifest['model'], revision=manifest['revision'],
                    torch_dtype=torch.float16, attn_implementation='sdpa' if method == 'ours' else 'eager').to('cuda:0').eval()
        model, updater = patch_model(model, method, config)
        # Avoid allocating all vocabulary logits during prefill in old HF paths.
        model.lm_head.register_forward_pre_hook(lambda module, inputs: (inputs[0][:, -1:, :],))
        archive = None
        if method == 'ours':
            from .phase_one_ours import PromptArchive
            archive = PromptArchive(model, manifest, path.parent / 'calibration')
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos] if eos is not None else []
        done = {r['id'] for r in state['rows']}
        for example in manifest['examples']:
            if example['id'] in done:
                continue
            torch.cuda.synchronize()
            start = time.perf_counter()
            ids, extra = generate(model, updater, method, example, config, manifest['seed'], eos, manifest['stop_ids'], archive)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            text = tokenizer.decode(ids, skip_special_tokens=True)
            append_row(path, state, dict(id=example['id'], text=text, generated_ids=ids,
                        output_tokens=len(ids), elapsed_seconds=elapsed,
                        **score_generation(example['data'], text, manifest['benchmark']), **extra))
            print(f'{method}: {len(state["rows"])}/{len(state["expected"])} saved', flush=True)
    except Exception as error:
        state.update(status='failed', error=str(error), traceback=traceback.format_exc())
        atomic_json(path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'worker', 'report', 'repair-rocket'))
    parser.add_argument('--out', type=Path, default=Path('outputs/kaggle-t4'))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=[MODELS[0]])
    parser.add_argument('--methods', nargs='+', choices=tuple(LABELS), default=list(LABELS))
    parser.add_argument('--benchmarks', nargs='+', choices=('longbenchv2', 'longgenbench'), default=['longbenchv2', 'longgenbench'])
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--prompt-cap', type=int, default=2048)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--env-root', type=Path, default=Path('.envs'))
    parser.add_argument('--manifest-index', type=int)
    parser.add_argument('--method', choices=tuple(LABELS))
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.action == 'worker':
        worker(args)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.out / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another launcher is using this output directory')
        if args.action == 'report':
            write_report(args.out, json.loads((args.out / 'inputs.json').read_text(encoding='utf-8')))
            return
        if args.action == 'repair-rocket':
            frozen = json.loads((args.out / 'inputs.json').read_text(encoding='utf-8'))
            if frozen.get('config', {}).get('profile') != PROFILE or 'rocketkv' not in frozen['config']['methods']:
                raise ValueError('This is not a compatible T4 output directory containing RocketKV')
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('Select a Kaggle GPU accelerator before repairing RocketKV')
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
            failure_path = args.out / 'worker_failures.json'
            worker_failures = json.loads(failure_path.read_text(encoding='utf-8')) if failure_path.exists() else {}
            repair_path = args.out / 'repairs' / 'rocket-fp32.json'
            repair = json.loads(repair_path.read_text(encoding='utf-8')) if repair_path.exists() else {
                'schema': 'kaggle-t4-rocket-fp32-repair-v1',
                'reason': 'RocketKV FP16 retrieval scores produced non-finite LongGenBench logits on Tesla T4.',
                'policy': 'Archive and recompute every RocketKV benchmark row; preserve all other completed methods.',
                'archived': [], 'jobs': []}
            failures = []
            # Archive both old RocketKV benchmark results first so the numerical
            # implementation stays consistent across the repaired comparison.
            for manifest in frozen['manifests']:
                path = result_path(args.out, manifest, 'rocketkv')
                old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                if path.exists() and 'FP32 retrieval attention' not in old.get('implementation', ''):
                    destination = args.out / 'repairs' / 'pre-fp32-rocket' / path.relative_to(args.out)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise FileExistsError(f'Repair archive already exists but current result is old: {destination}')
                    path.replace(destination)
                    log_path, log_destination = path.with_suffix('.log'), destination.with_suffix('.log')
                    if log_path.exists():
                        log_path.replace(log_destination)
                    repair['archived'].append(dict(result=str(destination.relative_to(args.out)),
                                                   prior_status=old.get('status'),
                                                   prior_run_identity=old.get('run_identity')))
            atomic_json(repair_path, repair)
            for index, manifest in enumerate(frozen['manifests']):
                path = result_path(args.out, manifest, 'rocketkv')
                path.parent.mkdir(parents=True, exist_ok=True)
                executable = args.env_root.resolve() / 't4-baselines' / 'bin/python'
                command = [str(executable), '-u', '-m', 'experiments.kaggle_t4', 'worker',
                           '--out', str(args.out), '--manifest-index', str(index), '--method', 'rocketkv']
                print(f'Repairing {manifest["model"]} {manifest["benchmark"]} RocketKV', flush=True)
                with path.with_suffix('.log').open('a', encoding='utf-8') as log:
                    result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible),
                                            stdout=log, stderr=subprocess.STDOUT)
                key = str(path.relative_to(args.out))
                if result.returncode:
                    failures.append(key)
                    worker_failures[key] = 'FP32 RocketKV repair worker failed; see result.log'
                    print_failure(path)
                else:
                    worker_failures.pop(key, None)
                repair['jobs'].append(dict(result=key, returncode=result.returncode))
                atomic_json(failure_path, worker_failures)
                atomic_json(repair_path, repair)
                write_report(args.out, frozen)
            if failures:
                raise SystemExit(f'{len(failures)} RocketKV repair jobs failed; see the log tail above.')
            print('RocketKV repair completed. See', args.out / 'comparison.md', flush=True)
            return
        cfg = protocol(args)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Select a Kaggle GPU accelerator before running')
        print('Sequential single-GPU jobs on:', torch.cuda.get_device_name(0), flush=True)
        frozen = prepare(args.out, cfg)
        failures = []
        failure_path = args.out / 'worker_failures.json'
        worker_failures = json.loads(failure_path.read_text(encoding='utf-8')) if failure_path.exists() else {}
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        for index, manifest in enumerate(frozen['manifests']):
            for method in cfg['methods']:
                env_name = 't4-ours' if method == 'ours' else 't4-baselines'
                executable = args.env_root.resolve() / env_name / 'bin/python'
                path = result_path(args.out, manifest, method)
                path.parent.mkdir(parents=True, exist_ok=True)
                command = [str(executable), '-u', '-m', 'experiments.kaggle_t4', 'worker',
                           '--out', str(args.out), '--manifest-index', str(index), '--method', method]
                print(f'Running {manifest["model"]} {manifest["benchmark"]} {method}; log: {path.with_suffix(".log")}', flush=True)
                with path.with_suffix('.log').open('a', encoding='utf-8') as log:
                    result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible),
                                            stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    failures.append(str(path))
                    # Never damage a valid checkpoint on an identity/preflight error.
                    worker_failures[str(path.relative_to(args.out))] = 'Worker failed; see result.log'
                    print_failure(path)
                else:
                    worker_failures.pop(str(path.relative_to(args.out)), None)
                atomic_json(failure_path, worker_failures)
                write_report(args.out, frozen)
        if failures:
            raise SystemExit(f'{len(failures)} jobs failed. Inspect result.log files; rerun the same command to resume.')
        print('Completed. See', args.out / 'comparison.md', flush=True)


if __name__ == '__main__':
    main()
