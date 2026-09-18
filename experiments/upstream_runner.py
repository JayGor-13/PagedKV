"""Isolated official ArkVale/KIVI and full-cache quality/system measurements.

Run with each method's own Python environment. No model download or CUDA import
is needed to import compatibility checks, create a plan, or collect reports.
"""
import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

from .benchmark_state import ROOT, digest, source_identity, atomic_json, open_checkpoint, append_row


class Unsupported(ValueError):
    pass


def check_compatibility(method, config, settings, transformers_version):
    architecture = config['model_type']
    if method == 'arkvale':
        if transformers_version != '4.40.0':
            raise Unsupported('ArkVale requires its separate Transformers 4.40.0 environment')
        if architecture != 'llama':
            raise Unsupported('Unmodified ArkVale adapter is validated here only for Llama; no Qwen port')
        rope = config.get('rope_scaling') or {}
        if rope and rope.get('rope_type', rope.get('type')) != 'linear':
            raise Unsupported('ArkVale custom RoPE does not implement Llama-3.1/dynamic/YaRN scaling')
        if settings['page_size'] not in (16, 32):
            raise Unsupported('Pinned ArkVale kernels compile page sizes 16 and 32 only')
        head_dim = config.get('head_dim') or config['hidden_size'] // config['num_attention_heads']
        group = config['num_attention_heads'] // config['num_key_value_heads']
        if head_dim != 128 or group not in (1, 4, 8):
            raise Unsupported('Pinned ArkVale dispatch requires head_dim=128 and GQA group 1/4/8')
    elif method == 'kivi':
        if transformers_version != '4.43.1':
            raise Unsupported('KIVI requires its separate Transformers 4.43.1 environment')
        if architecture != 'llama':
            raise Unsupported('This official KIVI bridge supports Llama; Qwen needs an architecture port')
        if settings['group_size'] not in (32, 64, 128) or settings['residual_length'] % settings['group_size']:
            raise Unsupported('KIVI requires group 32/64/128 and residual length divisible by group')
        head_dim = config.get('head_dim') or config['hidden_size'] // config['num_attention_heads']
        if head_dim % settings['group_size']:
            raise Unsupported('KIVI value quantization group must divide head dimension')
    elif method != 'full':
        raise Unsupported(f'Unknown method: {method}')


def quality_cases(data):
    cases = []
    for doc in data['documents']:
        if len(doc['token_ids']) < 2:
            raise ValueError('document needs at least two tokens for upstream prefill')
        for qi, q in enumerate(doc['questions']):
            if not q['token_ids']:
                raise ValueError('empty question')
            cases.append((f"{doc['id']}:{qi}", doc, q))
    return cases


def load_model(args, cfg, prompt_length):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    dtype = torch.float16  # Identical precision across all official comparison arms.
    extra = dict(config=config, revision=args.revision, torch_dtype=dtype,
                 device_map='cuda:0', attn_implementation=args.attention)
    if args.gpus > 1:
        if args.method != 'full':
            raise Unsupported('Official ArkVale/KIVI bridges currently require one GPU')
        from .gpu_layout import model_load_kwargs
        extra.update(model_load_kwargs(args.gpus, args.gpu_memory_gib))
    if args.method == 'kivi':
        sys.path.insert(0, str(ROOT / 'external/KIVI'))
        from models.llama_kivi import LlamaForCausalLM_KIVI
        config.k_bits = args.k_bits
        config.v_bits = args.v_bits
        config.group_size = args.group_size
        config.residual_length = args.residual_length
        config.use_flash = True
        # The pinned custom model still checks this legacy flag before creating
        # an otherwise quadratic dense causal mask. Inputs here have no padding;
        # prefill is causal FA2 and every continuation call is a single token.
        config._flash_attn_2_enabled = True
        extra.pop('attn_implementation')
        model = LlamaForCausalLM_KIVI.from_pretrained(args.model, **extra).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **extra).eval()
    if args.method == 'arkvale':
        from arkvale import adapter
        # Fixed capacity across examples; declare actual pages, not a byte ratio.
        pages = args.page_budget or max(8, math.ceil(prompt_length * args.budget_fraction / args.page_size))
        topk = args.page_topk or max(4, pages // 2)
        if not 3 < topk < pages:
            raise ValueError('ArkVale requires 3 < page_topk < page_budget')
        adapter.enable_arkvale(model, dtype=dtype, device=torch.device('cuda:0'),
            page_size=args.page_size, page_budgets=pages, page_topks=topk,
            n_sink_pages=2, n_win_pages=2, n_unlimited_layers=0,
            n_max_bytes=args.gpu_pool_gib * (1 << 30),
            n_max_cpu_bytes=args.cpu_pool_gib * (1 << 30))
        args.actual_page_budget = pages
        args.actual_page_topk = topk
    elif getattr(config, 'pretraining_tp', 1) == 1:
        # Upstream KIVI otherwise materializes [batch, context, vocabulary] logits.
        # This changes only the unused prompt logits, never attention/cache math.
        old_forward = model.lm_head.forward
        model.lm_head.forward = lambda x: old_forward(x[:, -1:, :])
    else:
        raise Unsupported('pretraining_tp != 1 is not supported by last-logit memory control')
    from .gpu_layout import assert_gpu_only
    assert_gpu_only(model)
    return model


def forward(model, ids, cache, method):
    if method == 'arkvale':
        # ArkVale owns its cache and returns the sentinel "dummy"; never pass it
        # to HF. q_len > 1 resets ArkVale state, so all continuation calls are 1 token.
        return model(input_ids=ids, use_cache=False, return_dict=True)
    return model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)


def generate(model, document, question, n_new, batch, method, eos_ids=()):
    """Document prefill then question tokens; TTFT includes both and first argmax.

    Each decode measurement includes argmax and CUDA synchronization. Batch rows
    are replicated prompts, deliberately labeled a synthetic throughput workload.
    """
    import torch
    def tensor(ids):
        return torch.tensor(ids, device='cuda:0', dtype=torch.long)[None].repeat(batch, 1)
    ids = tensor(document)
    suffix = [tensor([token]) for token in question]
    synchronize()
    start = time.perf_counter()
    result = forward(model, ids, None, method)
    cache = result.past_key_values
    for token in suffix:
        result = forward(model, token, cache, method)
        cache = result.past_key_values
    token = result.logits[:, -1].argmax(-1, keepdim=True)
    generated = [int(token[0, 0])]
    synchronize()
    ttft = time.perf_counter() - start
    steps = []
    for _ in range(n_new - 1):
        if generated[-1] in eos_ids:
            break
        start = time.perf_counter()
        result = forward(model, token, cache, method)
        cache = result.past_key_values
        token = result.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(int(token[0, 0]))
        synchronize()
        steps.append(time.perf_counter() - start)
    total = ttft + sum(steps)
    return generated, dict(ttft_s=ttft, decode_ms_per_token=1000 * sum(steps) / len(steps) if steps else None,
        decode_tokens_per_s=batch * len(steps) / sum(steps) if steps else None,
        end_to_end_tokens_per_s=batch * len(generated) / total, elapsed_s=total,
        decode_steps_s=steps, generated_tokens_per_sequence=len(generated), batch_size=batch,
        peak_gpu_allocated_gb=sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9,
        peak_gpu_reserved_gb=sum(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())) / 1e9,
        per_gpu_peak_allocated_gb=[torch.cuda.max_memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())],
        bandwidth_utilization_pct=None,
        bandwidth_note='not measured; requires hardware profiler DRAM counters',
        memory_scope='PyTorch allocator including model, cache pools, prefill and decode; excludes external CUDA allocations')


def synchronize():
    import torch
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def reset_memory():
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('full', 'arkvale', 'kivi'), required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--kind', choices=('quality', 'system'), default='quality')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--attention', choices=('sdpa', 'flash_attention_2'), default='flash_attention_2')
    p.add_argument('--new-tokens', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--gpu-memory-gib', type=int, default=120)
    p.add_argument('--context', type=int, default=65536)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--budget-fraction', type=float, default=.125)
    p.add_argument('--page-size', type=int, default=32)
    p.add_argument('--page-budget', type=int, default=0)
    p.add_argument('--page-topk', type=int, default=0)
    p.add_argument('--gpu-pool-gib', type=int, default=40)
    p.add_argument('--cpu-pool-gib', type=int, default=80)
    p.add_argument('--k-bits', type=int, choices=(2, 4), default=2)
    p.add_argument('--v-bits', type=int, choices=(2, 4), default=2)
    p.add_argument('--group-size', type=int, default=32)
    p.add_argument('--residual-length', type=int, default=128)
    return p


def run(args):
    if not re.fullmatch(r'[0-9a-fA-F]{40}', args.revision):
        raise ValueError('--revision must be an immutable 40-character model commit')
    if min(args.new_tokens, args.batch_size, args.context, args.repeats, args.page_size,
           args.gpu_pool_gib, args.cpu_pool_gib, args.group_size, args.residual_length) < 1:
        raise ValueError('sizes must be positive')
    if args.warmup < 0 or args.page_budget < 0 or args.page_topk < 0 or not 0 < args.budget_fraction <= 1:
        raise ValueError('invalid warmup/budget')
    data = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if (data['model'], data.get('revision')) != (args.model, args.revision):
        raise ValueError('model/revision must match frozen manifest')
    cases = quality_cases(data)
    if not cases:
        raise ValueError('empty manifest')
    expected = [c[0] for c in cases] if args.kind == 'quality' else [f'repeat:{i}' for i in range(args.repeats)]
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'transformers')}
    settings = {k: v for k, v in vars(args).items() if k not in ('out', 'resume')}
    metadata = dict(method=args.method, kind=args.kind, config=settings, versions=versions,
                    model=args.model, revision=args.revision, manifest_sha256=digest(data),
                    task=data.get('task', data.get('protocol')), protocol='document_prefill_question_tokenwise_v1',
                    source_hashes=source_identity(), dataset_metadata={k: data.get(k) for k in
                        ('complete_task_split', 'truncation', 'truncated_documents', 'prompt_protocol')})
    metadata['upstream_lock'] = json.loads((ROOT / 'configs/upstream_baselines.json').read_text())
    metadata['attention_backend'] = {'full': args.attention, 'kivi': 'FA2_prefill_KIVI_quantized_decode',
                                     'arkvale': 'ArkVale_FlashInfer'}[args.method]
    identity = digest(metadata)
    state = open_checkpoint(args.out, identity, expected, metadata, args.resume)
    if state['status'] == 'completed':
        if args.method != 'full':
            from scripts.fetch_baselines import verify
            if state.get('upstream') != verify(args.method):
                raise ValueError('upstream changed since completed checkpoint')
        print(f'Already complete: {args.out}')
        return
    atomic_json(args.out, state)
    try:
        from huggingface_hub import hf_hub_download
        # Read raw JSON first: old AutoConfig would fail before explaining new RoPE.
        if Path(args.model).is_dir():
            cfg = json.loads((Path(args.model) / 'config.json').read_text())
            raise Unsupported('Use a pinned Hub model ID for reproducible runs, not an unversioned local model')
        cfg_path = hf_hub_download(args.model, 'config.json', revision=args.revision)
        cfg = json.loads(Path(cfg_path).read_text())
        check_compatibility(args.method, cfg, settings, versions['transformers'])
        if any(q.get('metric') == 'longbench' for _, _, q in cases):
            from .benchmark_scores import longbench_metrics
            longbench_metrics()  # Fail missing scorer dependencies before loading weights.
        if args.method != 'full':
            from scripts.fetch_baselines import verify
            state['upstream'] = verify(args.method)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; execute this job on the H200')
        if torch.cuda.device_count() != args.gpus:
            raise ValueError('CUDA_VISIBLE_DEVICES must expose exactly --gpus devices')
        device = dict(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      total_bytes=torch.cuda.get_device_properties(0).total_memory,
                      cuda=torch.version.cuda, python=platform.python_version())
        if state.get('device') and state['device'] != device:
            raise ValueError('resume GPU/runtime mismatch')
        state['device'] = device
        state['status'] = 'running'
        torch.manual_seed(args.seed)
        max_prompt = max(len(d['token_ids']) + len(q['token_ids']) for _, d, q in cases)
        max_total = max(len(d['token_ids']) + len(q['token_ids']) + q.get('max_new_tokens', args.new_tokens)
                        for _, d, q in cases)
        if args.kind == 'system':
            max_prompt = args.context
            max_total = args.context + args.new_tokens
        if max_total > cfg['max_position_embeddings']:
            raise Unsupported(f'{max_total} tokens exceeds model context {cfg["max_position_embeddings"]}; no automatic RoPE extension')
        model = load_model(args, cfg, max_prompt)
        state['effective_settings'] = vars(args).copy()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        if data.get('tokenizer_sha256') and hasattr(tok, 'backend_tokenizer'):
            if digest_tokenizer(tok) != data['tokenizer_sha256']:
                raise ValueError('frozen tokenizer differs from runtime tokenizer')
        eos = model.generation_config.eos_token_id
        eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
        done = {r['id'] for r in state['rows']}
        with torch.inference_mode():
            if args.kind == 'quality':
                from .benchmark_scores import score_question
                for uid, doc, q in cases:
                    if uid in done:
                        continue
                    reset_memory()
                    ids, telemetry = generate(model, doc['token_ids'], q['token_ids'],
                        q.get('max_new_tokens', args.new_tokens), 1, args.method, eos)
                    text = tok.decode(ids, skip_special_tokens=True)
                    refs = q.get('answers', [q['target']])
                    append_row(args.out, state, dict(id=uid, document=doc['id'], style=q['style'],
                        generated_ids=ids, generated_text=text, references=refs,
                        prompt_tokens=len(doc['token_ids']) + len(q['token_ids']),
                        **score_question(text, q), telemetry=telemetry))
                    print(f'{args.method}: {len(state["rows"])}/{len(expected)} saved', flush=True)
            else:
                base = cases[0][1]['token_ids']
                # Synthetic repeat creates exactly context tokens, independent of tokenized dataset length.
                prompt = (base * math.ceil(args.context / len(base)))[:args.context]
                state['workload'] = 'synthetic_repeated_document_equal_length_replicated_batch_fixed_output'
                for _ in range(args.warmup):
                    generate(model, prompt, [], args.new_tokens, args.batch_size, args.method)
                for uid in expected:
                    if uid in done:
                        continue
                    reset_memory()
                    _, telemetry = generate(model, prompt, [], args.new_tokens, args.batch_size, args.method)
                    append_row(args.out, state, dict(id=uid, context=args.context, **telemetry))
        state['status'] = 'completed'
        atomic_json(args.out, state)
    except Exception as error:
        state['status'] = 'unsupported' if isinstance(error, Unsupported) else 'failed'
        state['error'] = f'{type(error).__name__}: {error}'
        atomic_json(args.out, state)
        raise


def digest_tokenizer(tok):
    import hashlib
    return hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest()


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
