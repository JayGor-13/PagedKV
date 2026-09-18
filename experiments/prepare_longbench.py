"""Freeze complete English LongBench v1 QA splits for question-blind KV evaluation.

Reads the official data.zip without executing dataset loader code. This uses a
document-first prompt and English QA F1, not an official leaderboard reproduction.
No test-row subsampling or silent document truncation. Overlength rows fail.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from . import reference_task as D
from .benchmark_scores import LONGBENCH16

TASKS = ('narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique')
SOURCE = 'https://huggingface.co/datasets/zai-org/LongBench'


def prepare(tok, rows, model, task, max_context, calibration, source, truncate_middle=False, task_prompts=None):
    if task not in (LONGBENCH16 if task_prompts else TASKS) or max_context < 1:
        raise ValueError('unsupported task or invalid max-context')
    documents = []
    for i, row in enumerate(rows):
        context = row['context']
        answers = row['answers']
        if not context.strip() or not answers or any(not isinstance(a,str) or not a.strip() for a in answers):
            raise ValueError(f'invalid context/answers in row {i}')
        # Compression sees no question. The prompt split is identical across methods.
        prefix = 'Read the document below.\n\n' + context
        suffix = '\n\nAnswer the question using the document. Give only the answer.\nQuestion: ' + row['input'] + '\nAnswer:'
        new_tokens = 128
        if task_prompts:
            template, new_tokens = task_prompts[task]
            before, after = template.split('{context}')
            prefix = before.format(**row) + context
            suffix = after.format(**row)
        ids = tok(prefix, add_special_tokens=False).input_ids
        qids = tok(suffix, add_special_tokens=False).input_ids
        original_tokens=len(ids)
        truncated=False
        if len(ids)+len(qids)+new_tokens > max_context:
            if not truncate_middle:
                raise ValueError(f'{task} row {i}: {len(ids)+len(qids)+128} tokens including answer budget exceeds '
                             f'{max_context}; no rows were silently removed or truncated')
            budget=max_context-len(qids)-new_tokens
            if budget < 132:
                raise ValueError('question/answer budget leaves too little document context')
            left=(budget+1)//2
            ids=ids[:left]+ids[-(budget-left):]
            truncated=True
        documents.append(dict(id=i, source_id=row.get('_id',str(i)), token_ids=ids,
            original_document_tokens=original_tokens,context_truncated=truncated,
            questions=[dict(style=task, token_ids=qids, answers=answers, target=answers[0],
                            fact_span=None, max_new_tokens=new_tokens,
                            metric='longbench' if task_prompts else 'qa_f1', all_classes=row.get('all_classes', []))]))
    if not documents:
        raise ValueError('empty dataset')
    return dict(schema=2, model=model, protocol='longbench_v1_english_qa_document_first',
        task=task, source=source, document_count=len(documents), complete_task_split=True,
        truncation='explicit_middle' if truncate_middle else 'none',
        truncated_documents=sum(d['context_truncated'] for d in documents),
        prompt_protocol='official_task_templates_split_plain_text_not_chat' if task_prompts else 'document_first_plain_text_v1_not_official_chat_template',
        oracle_available=False, calibration_evaluation_text_overlap='not_audited; local calibration, external test sources',
        calibration=calibration, documents=documents)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True, help='Pinned model/tokenizer commit')
    p.add_argument('--tasks', default=','.join(TASKS))
    p.add_argument('--suite16', action='store_true', help='All 16 English/code tasks with upstream task templates and task-specific scorers')
    p.add_argument('--data-zip', help='Existing official data.zip for offline preparation')
    p.add_argument('--dataset-revision', help='Pinned dataset commit; required for a download')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--max-context', type=int, default=32768)
    p.add_argument('--truncate-middle', action='store_true',help='Explicitly shorten overlength contexts, retaining every test row; logged per row')
    p.add_argument('--calib-docs', type=int, default=12)
    p.add_argument('--calib-ctx', type=int, default=2048)
    args = p.parse_args()
    tasks = list(LONGBENCH16) if args.suite16 else args.tasks.split(',')
    task_prompts = None
    if args.suite16:
        from scripts.fetch_baselines import verify
        from .benchmark_state import ROOT
        verify('kivi')
        cfg_dir = ROOT / 'external/KIVI/config'
        prompts = json.loads((cfg_dir / 'dataset2prompt.json').read_text(encoding='utf-8'))
        lengths = json.loads((cfg_dir / 'dataset2maxlen.json').read_text())
        task_prompts = {t: (prompts[t], lengths[t]) for t in tasks}
    if not set(tasks) <= set(LONGBENCH16 if args.suite16 else TASKS) or len(tasks) != len(set(tasks)):
        p.error('choose unique supported English QA tasks')
    if min(args.calib_docs,args.calib_ctx) < 1:
        p.error('positive calibration sizes required')
    if args.data_zip:
        path = Path(args.data_zip)
    else:
        if not args.dataset_revision:
            p.error('--dataset-revision is required when downloading')
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download('zai-org/LongBench', 'data.zip', repo_type='dataset', revision=args.dataset_revision))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    calibration = [x.tolist() for x in D.load_corpus(tok,args.calib_docs,args.calib_ctx)]
    with path.open('rb') as f:
        archive_hash = hashlib.file_digest(f,'sha256').hexdigest()
    out = Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for task in tasks:
            dest = out/f'{task}.json'
            if dest.exists():
                raise FileExistsError(f'refusing to replace frozen manifest {dest}')
            members = [name for name in z.namelist() if Path(name).name == task+'.jsonl']
            if len(members) != 1:
                raise ValueError(f'expected exactly one {task}.jsonl in archive')
            with z.open(members[0]) as f:
                rows = [json.loads(line) for line in f if line.strip()]
            data = prepare(tok,rows,args.model,task,args.max_context,calibration,
                           dict(url=SOURCE,revision=args.dataset_revision,zip_sha256=archive_hash),args.truncate_middle,task_prompts)
            data['revision'] = args.revision
            data['tokenizer_sha256'] = hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest()
            dest.write_text(json.dumps(data,ensure_ascii=False)+'\n',encoding='utf-8')
            print(f'{task}: all {len(rows)} rows -> {dest}',flush=True)


if __name__ == '__main__':
    main()
