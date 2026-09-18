"""Resume the FreeKV LongGenBench local Qwen3-32B judge, one check at a time."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from .benchmark_state import atomic_json, digest, source_identity, open_checkpoint, append_row
from .freekv_protocol import judge_checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='outputs/phase-one')
    p.add_argument('--gpus', type=int, default=1, choices=(1, 2, 4))
    p.add_argument('--batch-checks', type=int, default=32)
    args = p.parse_args()
    if args.batch_checks < 1:
        p.error('--batch-checks must be positive')
    out = Path(args.out)
    import fcntl
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(out, args.gpus, args.batch_checks)


def run(out, gpus, batch_checks):
    from scripts.fetch_baselines import verify
    upstream = verify('freekv')
    revisions = json.loads((out / 'revisions.json').read_text())
    plan = json.loads((out / 'plan.json').read_text())
    llm = tokenizer = None
    for job in plan['jobs']:
        if job['benchmark'] != 'longgenbench' or not Path(job['output']).exists():
            continue
        generation = json.loads(Path(job['output']).read_text())
        if generation['status'] != 'completed' or generation['smoke']:
            continue
        checks = [c for row in generation['rows'] for c in judge_checks(row)]
        path = Path(job['output']).with_name('judge.json')
        metadata = dict(judge_model=revisions['judge_model'], judge_revision=revisions['judge_revision'],
                        generation_sha256=digest(generation), upstream=upstream,
                        vllm=importlib.metadata.version('vllm'), gpus=gpus,
                        config=dict(temperature=.95, top_p=.95, max_tokens=50, seed=42, enable_thinking=False),
                        scoring='FreeKV: yes substring; missing blocks omitted from accuracy denominator; completion separate')
        state = open_checkpoint(path, digest([metadata, source_identity()]), [c['id'] for c in checks] or ['__no_checks__'], metadata, True)
        if state['status'] == 'completed':
            continue
        if not checks:
            append_row(path, state, dict(id='__no_checks__', answer=None))
            continue
        if llm is None:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(revisions['judge_model'], revision=revisions['judge_revision'])
            llm = LLM(model=revisions['judge_model'], revision=revisions['judge_revision'],
                      tokenizer_revision=revisions['judge_revision'], dtype='bfloat16',
                      tensor_parallel_size=gpus, disable_custom_all_reduce=True, seed=42,
                      gpu_memory_utilization=.9)
        done = {r['id'] for r in state['rows']}
        pending = [c for c in checks if c['id'] not in done]
        try:
            for start in range(0, len(pending), batch_checks):
                batch = pending[start:start+batch_checks]
                texts = [tokenizer.apply_chat_template([dict(role='user', content=c['prompt'])],
                         tokenize=False, enable_thinking=False, add_generation_prompt=True) for c in batch]
                outputs = llm.generate(texts, SamplingParams(temperature=.95, top_p=.95, max_tokens=50, seed=42))
                for check, output in zip(batch, outputs):
                    response = output.outputs[0].text
                    append_row(path, state, dict(id=check['id'], category=check['category'], text=response,
                                                answer='yes' if 'yes' in response.strip().lower() else 'no'))
            print(f'Judged {job["name"]}: {len(checks)} checks', flush=True)
        except Exception as error:
            state.update(status='failed', error=str(error))
            atomic_json(path, state)
            raise
    from .phase_one import write_report
    write_report(plan, out)


if __name__ == '__main__':
    main()
