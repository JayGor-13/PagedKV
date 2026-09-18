"""Per-example resumable generation for frozen FreeKV-protocol manifests."""
import argparse
import importlib.metadata
import json
import os
import re
from pathlib import Path
import time
import traceback
from .benchmark_state import atomic_json, digest, source_identity, open_checkpoint, append_row
from .freekv_protocol import settings, sample_seed, score_generation
from .phase_one_backends import LABELS, load_model, reset, source_for


def sample(logits, temperature, top_p, generator):
    import torch
    # CPU sampling keeps each example's RNG independent of GPU assignment.
    scores = logits.detach().float().cpu()
    if temperature <= 0:
        return int(scores.argmax())
    probabilities = torch.softmax(scores / temperature, -1)
    if top_p < 1:
        sorted_p, indices = torch.sort(probabilities, descending=True)
        mask = sorted_p.cumsum(-1) - sorted_p > top_p
        sorted_p[mask] = 0
        probabilities = torch.zeros_like(probabilities).scatter_(0, indices, sorted_p)
    return int(torch.multinomial(probabilities, 1, generator=generator))


def generate(model, updater, method, example, config, seed, eos, stop, archive=None):
    import torch
    generator = torch.Generator(device='cpu').manual_seed(sample_seed(seed, example['id']))
    torch.manual_seed(sample_seed(seed, example['id']))
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([example['token_ids']], device=device)
    reset(model, updater, method, ids, config)
    extra = {}
    with torch.inference_mode():
        if archive is not None:
            result, extra = archive.prefill(example['token_ids'], config)
        else:
            kwargs = {}
            if method in ('h2o', 'snapkv'):
                from transformers import DynamicCache
                kwargs['past_key_values'] = DynamicCache()
            result = model(input_ids=ids, use_cache=True, **kwargs)
        cache = result.past_key_values
        generated = []
        for step in range(config['max_new_tokens']):
            token = sample(result.logits[0, -1], config['temperature'], config['top_p'], generator)
            generated.append(token)
            if updater is not None:
                updater.update(token)
            if token in eos or (stop and generated[-len(stop):] == stop):
                break
            if step + 1 == config['max_new_tokens']:
                break
            kwargs = dict(position_ids=torch.tensor([[ids.shape[1] + step]], device=device))
            if method == 'ours':
                kwargs['logits_to_keep'] = 1
            if method in ('h2o', 'snapkv'):
                kwargs['cache_position'] = torch.tensor([ids.shape[1] + step], device=device)
            result = model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                           use_cache=True, **kwargs)
            cache = result.past_key_values
    if updater is not None:
        extra.update(updater.finish())
    return generated, extra


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--method', choices=tuple(LABELS), required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--gpus', type=int, required=True)
    p.add_argument('--gpu-memory-gib', type=int, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if not re.fullmatch('[0-9a-f]{40}', manifest.get('revision', '')):
        raise ValueError('a frozen 40-character model revision is required')
    from scripts.fetch_baselines import verify
    # Includes protocol/judge source as well as the selected backend.
    upstream = {name: verify(name) for name in {'freekv', source_for(args.method)}}
    config = settings(manifest['benchmark'], args.method)
    if manifest['smoke']:
        config['max_new_tokens'] = 8
    packages = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate')}
    identity = digest([manifest, args.method, config, upstream, packages, source_identity(),
                       args.gpus, args.gpu_memory_gib])
    metadata = dict(model=manifest['model'], revision=manifest['revision'], benchmark=manifest['benchmark'],
                    method=args.method, implementation=LABELS[args.method], config=config,
                    smoke=manifest['smoke'], protocol=manifest['protocol'], upstream=upstream,
                    packages=packages, manifest_sha256=digest(manifest), gpu_count=args.gpus)
    state = open_checkpoint(args.out, identity, [e['id'] for e in manifest['examples']], metadata, args.resume)
    if state['status'] == 'completed':
        print('Already completed, identity verified:', args.out)
        return
    atomic_json(args.out, state)
    try:
        import torch
        if torch.cuda.device_count() != args.gpus:
            raise ValueError('CUDA_VISIBLE_DEVICES must contain exactly --gpus devices')
        state['hardware'] = [dict(name=torch.cuda.get_device_name(i), total_bytes=torch.cuda.get_device_properties(i).total_memory)
                             for i in range(args.gpus)]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(manifest['model'], revision=manifest['revision'], use_fast=False)
        model, updater = load_model(manifest, args.method, config, args.gpus, args.gpu_memory_gib)
        archive = None
        if args.method == 'ours':
            from .phase_one_ours import PromptArchive
            archive = PromptArchive(model, manifest, Path(args.out).parent / 'calibration')
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos] if eos is not None else []
        done = {r['id'] for r in state['rows']}
        for example in manifest['examples']:
            if example['id'] in done:
                continue
            for device in range(args.gpus):
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            ids, extra = generate(model, updater, args.method, example, config, manifest['seed'], eos,
                                  manifest['stop_ids'], archive)
            for device in range(args.gpus):
                torch.cuda.synchronize(device)
            text = tokenizer.decode(ids, skip_special_tokens=True)
            row = dict(id=example['id'], generated_ids=ids, text=text, output_tokens=len(ids),
                       elapsed_seconds=time.perf_counter()-start, telemetry_scope='quality_run_not_serving_benchmark',
                       **score_generation(example['data'], text, manifest['benchmark']), **extra)
            append_row(args.out, state, row)
            print(f'{args.method}: {len(state["rows"])}/{len(state["expected"])} committed', flush=True)
    except Exception as error:
        state.update(status='failed', error=str(error), traceback=traceback.format_exc())
        atomic_json(args.out, state)
        raise


if __name__ == '__main__':
    main()
