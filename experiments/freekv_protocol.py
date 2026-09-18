"""FreeKV evaluation protocol, without importing GPU code or executing upstream CLIs."""
import ast
import hashlib
import re
from .benchmark_state import ROOT

FREEKV = ROOT / 'external/FreeKV'


def upstream_judge_functions():
    # Import only three pure functions. Upstream eval.py loads a hard-coded GPU
    # model and parses sys.argv at module scope; importing it is not safe.
    path = FREEKV / 'accuracy/eval/LongGenBench/eval.py'
    names = {'parse_blocks', 'create_prompts', 'calculate_completion_rate'}
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('upstream judge API changed')
    namespace = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def settings(benchmark, method):
    longgen = benchmark == 'longgenbench'
    if benchmark not in ('longbenchv2', 'longgenbench'):
        raise ValueError(benchmark)
    return dict(max_new_tokens=16000 if longgen else 128,
                temperature=.95 if longgen else 0., top_p=1.,
                sink=512 if longgen else 128, recent=512 if longgen else 128,
                budget=1024 if longgen else 1792, page_size=32, skip_layer=1,
                GQA_policy='maxS' if method == 'quest' else 'avgSM' if method == 'freekv' else 'avgS',
                spec_ret_steps=2, correct_sim=.9 if longgen else .8)


def format_prompt(item, benchmark):
    if benchmark == 'longgenbench':
        return item['prompt']
    prompt = (FREEKV / 'accuracy/eval/LongBench2/prompts/0shot.txt').read_text(encoding='utf-8')
    for key, field in [('DOC', 'context'), ('Q', 'question'), *[(f'C_{c}', f'choice_{c}') for c in 'ABCD']]:
        prompt = prompt.replace(f'${key}$', item[field].strip())
    return prompt


def chat_ids(tokenizer, prompt, model, benchmark, cap=120000):
    """Match FreeKV's truncate-before-chat protocol, including its re-tokenization."""
    ids = tokenizer.encode(prompt)
    original = len(ids)
    if benchmark == 'longbenchv2' and len(ids) > cap:
        half = cap // 2
        prompt = tokenizer.decode(ids[:half] + ids[-half:], skip_special_tokens=True)
    elif benchmark == 'longgenbench' and len(ids) >= cap:
        raise ValueError('LongGenBench prompt exceeds FreeKV cap')
    messages = [dict(role='user', content=prompt)]
    if 'qwen' in model.lower():
        messages.insert(0, dict(role='system', content='You are a helpful and harmless assistant. You are Qwen developed by Alibaba.'))
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = tokenizer.encode(rendered)
    else:
        result = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return result, dict(raw_prompt_tokens=original, truncated=original > cap,
                        prompt_tokens=len(result))


def extract_answer(text):
    text = text.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', text)
    if not match:
        match = re.search(r'The correct answer is ([A-D])', text)
    return match.group(1) if match else None


def sample_seed(seed, example_id):
    # Stable across ordering, failed jobs and restarts. This intentionally replaces
    # upstream's single advancing RNG stream and is recorded in the manifest.
    return (seed + int(hashlib.sha256(str(example_id).encode()).hexdigest()[:8], 16)) % (2**31)


def score_generation(item, text, benchmark):
    if benchmark == 'longbenchv2':
        answer = extract_answer(text)
        return dict(predicted_answer=answer, accuracy=float(answer == item['answer']),
                    difficulty=item.get('difficulty'), length=item.get('length'), domain=item.get('domain'))
    functions = upstream_judge_functions()
    blocks = (item['prefix'] + text).split('#*#')
    parsed = functions['parse_blocks'](blocks, item['type'])
    return dict(output_blocks=blocks, completion_rate=functions['calculate_completion_rate'](parsed, item['number']),
                judge_status='pending', **{k: item[k] for k in ('type', 'number', 'checks_once', 'checks_range', 'checks_periodic')})


def judge_checks(row):
    funcs = upstream_judge_functions()
    blocks = funcs['parse_blocks'](row['output_blocks'], row['type'])
    checks = []
    for category in ('once', 'range', 'periodic'):
        prompts, identifiers = funcs['create_prompts'](row[f'checks_{category}'], blocks)
        checks.extend(dict(id=f'{row["id"]}/{category}/{i}', category=category, prompt=p)
                      for i, p in zip(identifiers, prompts))
    return checks
