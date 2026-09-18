"""Freeze local RULER/MATH-500 JSONL exports without changing or subsampling rows.

RULER input: {input: complete_prompt, outputs: [reference_strings], index: id}.
MATH-500 input: {problem: question, answer: exact_reference}. MATH-500 is not a
long-context task unless a separate, documented long-context protocol is supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .benchmark_state import atomic_json
from . import reference_task as D


def prepare(tok, rows, benchmark, model, revision, source, max_context, calibration, new_tokens, tail=32):
    if tail < 1 or new_tokens < 1 or max_context < 2:
        raise ValueError('positive budgets required')
    docs = []
    for i, row in enumerate(rows):
        if benchmark == 'ruler':
            prompt, answers = row['input'], row['outputs']
            metric = 'ruler_reference_recall'
        elif benchmark == 'math500':
            prompt = 'Solve this problem. Put your final answer in \\boxed{}.\n\n' + row['problem'] + '\n\nSolution:'
            answers, metric = [row['answer']], 'math_boxed_em'
        else:
            raise ValueError('unknown benchmark')
        if not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError('nonempty string references required')
        ids = tok(prompt, add_special_tokens=False).input_ids
        if len(ids) < 3 or len(ids) + new_tokens > max_context:
            raise ValueError(f'row {i}: invalid length or context exceeded; no silent truncation')
        split = max(2, len(ids) - tail)
        docs.append(dict(id=i, source_id=row.get('index', str(i)), token_ids=ids[:split],
                         questions=[dict(style=benchmark, token_ids=ids[split:], answers=answers,
                                         target=answers[0], metric=metric, fact_span=None, max_new_tokens=new_tokens)]))
    if not docs:
        raise ValueError('empty input')
    return dict(schema=2, model=model, revision=revision, task=benchmark, protocol=benchmark + '_frozen_prompt_v1',
                prompt_protocol=f'complete_prompt_split_before_final_{tail}_tokens_not_question_blind',
                source=source, oracle_available=False, documents=docs, calibration=calibration,
                complete_task_split=False, note='All supplied rows retained; completeness of upstream split not asserted.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', choices=('ruler', 'math500'), required=True)
    p.add_argument('--input-jsonl', required=True)
    p.add_argument('--source-revision', required=True, help='Dataset/generator commit and configuration identity')
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-context', type=int, required=True)
    p.add_argument('--new-tokens', type=int, default=512)
    p.add_argument('--prompt-tail-tokens', type=int, default=32)
    args = p.parse_args()
    if Path(args.out).exists():
        raise FileExistsError('frozen manifest exists')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    raw = Path(args.input_jsonl).read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    data = prepare(tok, rows, args.benchmark, args.model, args.revision,
                   dict(revision=args.source_revision, sha256=hashlib.sha256(raw).hexdigest()), args.max_context,
                   [ids.tolist() for ids in D.load_corpus(tok, 12, 2048)], args.new_tokens, args.prompt_tail_tokens)
    if hasattr(tok, 'backend_tokenizer'):
        data['tokenizer_sha256'] = hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest()
    atomic_json(args.out, data)
    print(args.out)


if __name__ == '__main__':
    main()
