"""Run a small pretrained model on real LongBench v2 and LongGenBench inputs.

This CPU smoke test deliberately shortens prompts and generation. It checks Hub
authentication, frozen source revisions, model loading, inference, and scoring
wiring; its output is never a full benchmark or an H200 performance result.
"""
import argparse
import json
from pathlib import Path
import time

from experiments.benchmark_state import ROOT, atomic_json
from experiments.freekv_protocol import FREEKV, chat_ids, format_prompt, score_generation


class AuthenticationError(RuntimeError):
    """The requested authenticated Hub access is unavailable."""


def access_token(anonymous):
    from huggingface_hub import HfApi, get_token
    if anonymous:
        return False, None
    token = get_token()
    if not token:
        raise AuthenticationError('No Hugging Face token found. Set HF_TOKEN or log in; use --anonymous only for public models.')
    try:
        account = HfApi(token=token).whoami()['name']
    except Exception as error:
        if getattr(getattr(error, 'response', None), 'status_code', None) in (401, 403):
            raise AuthenticationError('Hugging Face rejected the token. Set a valid HF_TOKEN or log in again.') from error
        raise AuthenticationError('Could not validate the Hugging Face token; check network access and retry.') from error
    return token, account


def run(model_id, out, prompt_cap=512, max_new_tokens=4, anonymous=False,
        check_models=()):
    if prompt_cap < 64 or max_new_tokens < 1:
        raise ValueError('prompt cap must be at least 64 and generation must be positive')
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token, account = access_token(anonymous)
    api = HfApi(token=token)
    revisions = {}
    for name in dict.fromkeys((model_id, *check_models)):
        revision = api.model_info(name).sha
        try:
            hf_hub_download(name, 'config.json', revision=revision, token=token)
        except Exception as error:
            raise RuntimeError(f'Cannot read {name}/config.json; check this account\'s model access.') from error
        revisions[name] = revision

    lb_revision = api.dataset_info('THUDM/LongBench-v2').sha
    lb_rows = load_dataset('THUDM/LongBench-v2', revision=lb_revision,
                           split='train', token=token)
    if not lb_rows:
        raise RuntimeError('LongBench v2 train split is empty')
    lb = lb_rows[0]
    upstream = json.loads((ROOT / 'configs/upstream_baselines.json').read_text())['freekv']
    from scripts.fetch_baselines import verify
    verify('freekv')
    lg_path = FREEKV / 'accuracy/eval/LongGenBench/Dataset_short.json'
    lg_rows = json.loads(lg_path.read_text(encoding='utf-8'))
    if not lg_rows:
        raise RuntimeError('LongGenBench short dataset is empty')
    lg = lg_rows[0]

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revisions[model_id],
                                               use_fast=True, token=token)
    # The source LongBench item can exceed the model limit, but this smoke
    # truncates before inference. Suppress the tokenizer's premature warning.
    tokenizer.model_max_length = max(tokenizer.model_max_length, 1_000_000)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revisions[model_id], token=token,
        torch_dtype=torch.float32, attn_implementation='sdpa').eval()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rows = []
    for benchmark, item in (('longbenchv2', lb), ('longgenbench', lg)):
        # LongGen's full protocol forbids truncation. Here the short CPU check
        # explicitly uses the LongBench head/tail rule and records that choice.
        input_ids, stats = chat_ids(tokenizer, format_prompt(item, benchmark),
                                    model_id, 'longbenchv2', cap=prompt_cap)
        with torch.inference_mode():
            start = time.perf_counter()
            inputs = torch.tensor([input_ids])
            generated = model.generate(
                inputs, attention_mask=torch.ones_like(inputs),
                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.eos_token_id)[0, len(input_ids):].tolist()
            elapsed = time.perf_counter() - start
        response = tokenizer.decode(generated, skip_special_tokens=True)
        scoring = score_generation(item, response, benchmark)
        rows.append(dict(benchmark=benchmark, id=str(item.get('_id', 0)),
                         prompt_tokens=len(input_ids), source_prompt_tokens=stats['raw_prompt_tokens'],
                         truncated=stats['truncated'], generated_tokens=len(generated),
                         elapsed_seconds=elapsed, output=response,
                         scorer_fields=sorted(scoring)))
        print(f'{benchmark}: {len(input_ids)} prompt tokens, {len(generated)} output tokens', flush=True)

    report = dict(schema='real-hf-cpu-smoke-v1', run_kind='shortened_pretrained_wiring_only',
                  is_benchmark_result=False, device='cpu', model=model_id,
                  model_revision=revisions[model_id], checked_model_revisions=revisions,
                  dataset_revisions={'THUDM/LongBench-v2': lb_revision,
                                     'FreeKV/LongGenBench/Dataset_short.json': upstream['commit']},
                  dataset_row_counts={'longbenchv2': len(lb_rows),
                                      'longgenbench': len(lg_rows)},
                  authenticated_account=account, anonymous=anonymous,
                  prompt_cap=prompt_cap, max_new_tokens=max_new_tokens,
                  protocol_note='One real item from each dataset; LongGen prompt also shortened with head/tail; no research score',
                  rows=rows)
    atomic_json(out, report)
    print(f'Real-data CPU smoke completed: {Path(out).resolve()}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--out', default='outputs/real-hf-cpu-smoke.json')
    parser.add_argument('--prompt-cap', type=int, default=512)
    parser.add_argument('--max-new-tokens', type=int, default=4)
    parser.add_argument('--anonymous', action='store_true',
                        help='use only public Hub assets, ignoring a saved token')
    parser.add_argument('--check-models', nargs='*', default=[],
                        help='also verify config-file access for these model IDs')
    args = parser.parse_args()
    try:
        run(args.model, args.out, args.prompt_cap, args.max_new_tokens,
            args.anonymous, args.check_models)
    except AuthenticationError as error:
        parser.exit(1, f'Authentication error: {error}\n')


if __name__ == '__main__':
    main()
