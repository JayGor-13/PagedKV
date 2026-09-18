import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.benchmark_state import open_checkpoint, append_row, atomic_json, digest
from experiments.upstream_runner import check_compatibility, Unsupported, quality_cases
from experiments.gpu_layout import choose_layout
from experiments.h200_suite import make_plan, execute
from experiments.comparison_report import collect, write_reports
from experiments.benchmark_scores import score_question, boxed_answer
from experiments.prepare_benchmark import prepare


def test_atomic_resume_restarts_only_unfinished_work(tmp_path):
    path = tmp_path / 'run.json'
    meta = dict(method='full', kind='quality')
    state = open_checkpoint(path, 'identity', ['doc0', 'doc1'], meta)
    append_row(path, state, dict(id='doc0', score=.8))
    # A torn temporary file cannot overwrite the last committed checkpoint.
    path.with_suffix('.json.tmp').write_text('{broken')
    resumed = open_checkpoint(path, 'identity', ['doc0', 'doc1'], meta, resume=True)
    assert [r['id'] for r in resumed['rows']] == ['doc0']
    with pytest.raises(ValueError, match='identity'):
        open_checkpoint(path, 'different', ['doc0', 'doc1'], meta, resume=True)
    with pytest.raises(ValueError, match='duplicate'):
        append_row(path, resumed, dict(id='doc0'))
    append_row(path, resumed, dict(id='doc1', score=.9))
    assert json.loads(path.read_text())['status'] == 'completed'
    with pytest.raises(FileExistsError):
        open_checkpoint(path, 'identity', ['doc0', 'doc1'], meta)


def test_checkpoint_refuses_false_completion(tmp_path):
    path = tmp_path / 'bad.json'
    atomic_json(path, dict(run_identity='x', expected=['a', 'b'], status='completed', rows=[dict(id='a')]))
    with pytest.raises(ValueError, match='missing'):
        open_checkpoint(path, 'x', ['a', 'b'], {}, True)


def test_compatibility_never_silently_substitutes_models_or_rope():
    cfg = dict(model_type='llama', hidden_size=4096, num_attention_heads=32, num_key_value_heads=8)
    settings = dict(page_size=32, group_size=32, residual_length=128)
    check_compatibility('arkvale', cfg, settings, '4.40.0')
    check_compatibility('kivi', cfg, settings, '4.43.1')
    with pytest.raises(Unsupported, match='RoPE'):
        check_compatibility('arkvale', dict(cfg, rope_scaling={'rope_type': 'llama3'}), settings, '4.40.0')
    with pytest.raises(Unsupported, match='Qwen'):
        check_compatibility('kivi', dict(cfg, model_type='qwen2'), settings, '4.43.1')
    with pytest.raises(Unsupported, match='16 and 32'):
        check_compatibility('arkvale', cfg, dict(settings, page_size=64), '4.40.0')
    with pytest.raises(Unsupported, match='separate'):
        check_compatibility('kivi', cfg, settings, '5.16.1')


SPEC = dict(parameter_count_approx=14_700_000_000, layers=48, kv_heads=8, head_dim=128)


def test_gpu_count_policy_constrained_memory_and_no_implicit_quantization():
    assert choose_layout('full', SPEC, 1, 24, 16384)['status'] == 'insufficient_memory'
    layout = choose_layout('full', SPEC, 2, 24, 16384)
    assert layout['status'] == 'ready' and layout['layout'] == 'model_sharded' and layout['gpus'] == 2
    assert choose_layout('kivi', SPEC, 2, 24, 16384)['status'] == 'unsupported'
    with pytest.raises(ValueError):
        choose_layout('full', SPEC, 0, 130, 16384)


def manifest(tmp_path, model='meta-llama/Llama-3.1-8B-Instruct'):
    data = dict(model=model, revision='a' * 40, task='qasper', documents=[
        dict(id=0, token_ids=[1, 2, 3], questions=[dict(style='qasper', token_ids=[4], target='yes', max_new_tokens=8)])])
    path = tmp_path / (model.split('/')[-1] + '.json')
    path.write_text(json.dumps(data))
    return path, data


def test_plan_contains_requested_models_methods_and_unsupported_reasons(tmp_path):
    llama, _ = manifest(tmp_path)
    qwen7, _ = manifest(tmp_path, 'Qwen/Qwen2.5-7B-Instruct')
    qwen14, _ = manifest(tmp_path, 'Qwen/Qwen2.5-14B-Instruct')
    env = dict.fromkeys(('ours', 'full', 'kivi', 'arkvale'), '/env/python')
    plan = make_plan([llama, qwen7, qwen14], ['ours', 'full', 'kivi', 'arkvale', 'quest'], env, 2, 130, tmp_path / 'runs')
    assert len(plan['jobs']) == 30
    by_name = {j['name']: j for j in plan['jobs']}
    assert len(by_name) == 30
    assert any(j['method'] == 'kivi' and j['status'] == 'planned' for j in plan['jobs'])
    assert all(j['status'] == 'unsupported' for j in plan['jobs'] if j['method'] == 'arkvale')
    assert all(j['status'] == 'unsupported' for j in plan['jobs'] if j['method'] == 'ours' and j['kind'] == 'system')
    qwen_jobs = [j for j in plan['jobs'] if 'Qwen' in j['name'] and j['method'] == 'full' and j['kind'] == 'quality']
    assert len(qwen_jobs) == 2
    assert all('--gpus' in j['argv'] and j['layout']['gpus'] == 1 for j in qwen_jobs)
    assert all(j['argv'][-1] != '--execute' for j in plan['jobs'] if j['argv'])
    write_reports(plan, tmp_path / 'tables')
    assert 'unsupported' in (tmp_path / 'tables/comparison.md').read_text()


def test_page_sweep_holds_recall_tokens_fixed(tmp_path):
    path, _ = manifest(tmp_path)
    plan = make_plan([path], ['ours'], {'ours': 'python'}, 1, 130, tmp_path, ('quality',), ablations=True)
    for job in plan['jobs']:
        argv = job['argv']
        assert int(argv[argv.index('--page')+1]) * int(argv[argv.index('--recall-k')+1]) == 1024
    assert len(plan['jobs']) == 4


def test_report_does_not_average_partial_or_mix_configuration(tmp_path):
    path, data = manifest(tmp_path)
    plan = make_plan([path], ['full'], {'full': 'python'}, 1, 130, tmp_path / 'runs', ('quality',))
    job = plan['jobs'][0]
    state = dict(schema='h200-benchmark-v1', status='running', rows=[dict(id='0:0', style='qasper', token_f1=.8)],
                 expected=['0:0', '1:0'], config={}, manifest_sha256=digest(data), model=data['model'])
    atomic_json(job['output'], state)
    result = collect(plan)
    assert result['quality'][0]['score_pct'] is None
    state.update(status='completed', expected=['0:0'])
    atomic_json(job['output'], state)
    assert collect(plan)['quality'][0]['score_pct'] == 80
    job['returncode'] = 1
    assert collect(plan)['quality'][0]['score_pct'] is None
    state['manifest_sha256'] = 'wrong'
    atomic_json(job['output'], state)
    with pytest.raises(ValueError, match='manifest'):
        collect(plan)


def test_task_scores_are_not_qa_f1_aliases():
    q = dict(target='12', answers=['12', '34'], metric='ruler_reference_recall')
    assert score_question('12 only', q)['score'] == .5
    q = dict(target='\\frac{1}{2}', metric='math_boxed_em')
    assert score_question('Proof... \\boxed{\\frac{1}{2}}', q)['score'] == 1
    assert score_question('\\boxed{0.5}', q)['score'] == 0  # exact, not symbolic equivalence
    assert boxed_answer('\\boxed{1} and \\boxed{2}') == '2'


def test_external_dataset_freeze_keeps_rows_and_rejects_truncation():
    tok = lambda s, **kw: SimpleNamespace(input_ids=list(range(len(s))))
    rows = [dict(input='A long enough prompt to split', outputs=['123'])] * 2
    result = prepare(tok, rows, 'ruler', 'm', 'r', {}, 100, [[1, 2]], 10, tail=5)
    assert len(result['documents']) == 2
    assert len(quality_cases(result)) == 2
    with pytest.raises(ValueError, match='exceeded'):
        prepare(tok, rows, 'ruler', 'm', 'r', {}, 10, [[1, 2]], 10)


def test_scheduler_respects_visible_gpu_ids_and_continues_after_failure(tmp_path, monkeypatch):
    path, data = manifest(tmp_path)
    plan = make_plan([path], ['full', 'kivi'], {'full': 'python', 'kivi': 'python'}, 2, 130,
                     tmp_path / 'runs', ('quality',))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,5')
    calls = []
    def fake_run(argv, cwd, env, stdout, stderr):
        calls.append(env['CUDA_VISIBLE_DEVICES'])
        if argv[argv.index('--method')+1] == 'full':
            return SimpleNamespace(returncode=1)
        dest = argv[argv.index('--out')+1]
        atomic_json(dest, dict(schema='h200-benchmark-v1', status='completed',
            manifest_sha256=digest(data), config={}, rows=[dict(id='0:0', style='qasper', token_f1=.5)], expected=['0:0']))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('experiments.h200_suite.subprocess.run', fake_run)
    execute(plan, tmp_path / 'plan.json')
    assert sorted(calls) == ['3', '5']
    assert [j['status'] for j in plan['jobs']] == ['failed', 'completed']
    assert (tmp_path / 'comparison/comparison.md').exists()


def test_longbench_summary_templates_and_output_budgets():
    from experiments.prepare_longbench import prepare as prepare_lb
    tok = lambda s, **kw: SimpleNamespace(input_ids=[ord(c) for c in s])
    data = prepare_lb(tok, [dict(context='report body', input='', answers=['summary'])], 'm',
                      'gov_report', 600, [[1]], {}, task_prompts={'gov_report': ('Report: {context}\nSummary:', 512)})
    q = data['documents'][0]['questions'][0]
    assert q['metric'] == 'longbench' and q['max_new_tokens'] == 512
    with pytest.raises(ValueError, match='exceeds'):
        prepare_lb(tok, [dict(context='report body', input='', answers=['summary'])], 'm',
                   'gov_report', 50, [[1]], {}, task_prompts={'gov_report': ('Report: {context}\nSummary:', 512)})


def test_generation_token_count_eos_and_batch_throughput_on_cpu(monkeypatch):
    import torch
    from experiments.upstream_runner import generate
    original_tensor = torch.tensor
    def cpu_tensor(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return original_tensor(*args, **kwargs)
    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: 1000)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *a: 2000)
    calls = []
    def model(input_ids, past_key_values, **kwargs):
        calls.append(input_ids.shape[1])
        length = (past_key_values or 0) + input_ids.shape[1]
        logits = torch.zeros(input_ids.shape[0], 1, 16)
        logits[:, :, length] = 1
        return SimpleNamespace(logits=logits, past_key_values=length)
    ids, timing = generate(model, [1, 2, 3], [4], 3, 2, 'full')
    assert ids == [4, 5, 6] and calls == [3, 1, 1, 1]
    assert timing['generated_tokens_per_sequence'] == 3
    assert timing['decode_tokens_per_s'] == pytest.approx(4 / sum(timing['decode_steps_s']))
    calls.clear()
    ids, timing = generate(model, [1, 2, 3], [4], 3, 2, 'full', eos_ids=[4])
    assert ids == [4] and calls == [3, 1]
    assert timing['decode_ms_per_token'] is None
