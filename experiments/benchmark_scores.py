"""Task-specific scores. Keep metric names/protocols explicit in result tables."""
from functools import lru_cache
import importlib.util
import re

from .benchmark_state import ROOT
from .metrics import answer_scores

LONGBENCH16 = ('narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique',
              'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa', 'samsum',
              'passage_count', 'passage_retrieval_en', 'lcc', 'repobench-p')


@lru_cache(None)
def longbench_metrics():
    # Use the actual published scorer included in the pinned MIT-licensed KIVI
    # checkout rather than substituting QA F1 for summaries/code/classification.
    from scripts.fetch_baselines import verify
    verify('kivi')
    path = ROOT / 'external/KIVI/metrics.py'
    spec = importlib.util.spec_from_file_location('official_longbench_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boxed_answer(text):
    start = text.rfind('\\boxed{')
    if start < 0:
        return text.strip()
    start += len('\\boxed{')
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == '{') - (text[end] == '}')
        if depth == 0:
            return text[start:end].strip()
    return ''


def score_question(prediction, question):
    refs = question.get('answers', [question.get('target', '')])
    scores = answer_scores(prediction, refs)
    protocol = question.get('metric', 'qa_f1')
    if protocol == 'longbench':
        task = question['style']
        if task not in LONGBENCH16:
            raise ValueError('unsupported LongBench task')
        names = {**{t: 'qa_f1_score' for t in LONGBENCH16[:6]}, 'triviaqa': 'qa_f1_score',
                 **{t: 'rouge_score' for t in ('gov_report', 'qmsum', 'multi_news', 'samsum')},
                 'trec': 'classification_score', 'passage_count': 'count_score',
                 'passage_retrieval_en': 'retrieval_score', 'lcc': 'code_sim_score', 'repobench-p': 'code_sim_score'}
        name = names[task]
        pred = prediction.lstrip('\n').split('\n')[0] if task in ('trec', 'triviaqa', 'samsum') else prediction
        score = max(getattr(longbench_metrics(), name)(pred, ref, all_classes=question.get('all_classes', [])) for ref in refs)
        metric = 'longbench_' + name.removesuffix('_score')
    elif protocol == 'ruler_reference_recall':
        # Explicitly a reference-string recall, not RULER's effective-context-length aggregate.
        score = sum(ref.lower() in prediction.lower() for ref in refs) / len(refs)
        metric = protocol
    elif protocol == 'math_boxed_em':
        # Exact answer comparison; deliberately does not pretend to be symbolic equivalence.
        norm = lambda s: re.sub(r'\s+', '', boxed_answer(s)).strip('$')
        score = float(any(norm(prediction) == norm(ref) for ref in refs))
        metric = protocol
    elif protocol == 'qa_f1':
        score, metric = scores['token_f1'], 'qa_f1'
    else:
        raise ValueError(f'unknown metric: {protocol}')
    return dict(**scores, score=score, metric=metric)
