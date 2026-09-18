import json
from pathlib import Path
import pytest
import torch
from experiments.benchmark_state import atomic_json, digest
from experiments.freekv_protocol import settings, extract_answer, chat_ids, score_generation, judge_checks, sample_seed
from experiments.phase_one import make_plan, write_report, execute
from experiments.phase_one_worker import sample


def test_freekv_protocol_scoring_and_sampling():
    assert settings('longgenbench', 'freekv')['max_new_tokens'] == 16000
    assert settings('longgenbench', 'freekv')['correct_sim'] == .9
    assert settings('longbenchv2', 'quest')['GQA_policy'] == 'maxS'
    assert settings('longbenchv2', 'freekv')['budget'] == 1792
    assert extract_answer('**The correct answer is (C)**') == 'C'
    assert extract_answer('C') is None  # no broader scorer than FreeKV
    logits = torch.tensor([1., 2., 3.])
    def draw():
        gen = torch.Generator().manual_seed(sample_seed(42, 'stable-id'))
        return [sample(logits, .95, 1, gen) for _ in range(20)]
    assert draw() == draw()
    assert sample(logits, 0, 1, None) == 2


def test_longgen_uses_pinned_judge_and_missing_block_denominators():
    data = dict(prefix='#*# Week 1 ', type='Week', number=3,
                checks_once={'1': 'birthday', '3': 'party'}, checks_range={}, checks_periodic={})
    result = dict(id='sample', **score_generation(data, 'birthday #*# Week 2 weather', 'longgenbench'))
    assert result['completion_rate'] == pytest.approx(200/3)
    checks = judge_checks(result)
    assert len(checks) == 1 and checks[0]['id'] == 'sample/once/1'
    assert 'Example 1:' in checks[0]['prompt']
    assert result['judge_status'] == 'pending'


def test_truncation_precedes_chat():
    class Tokenizer:
        def encode(self, s):
            return list(map(ord, s))
        def decode(self, ids, **kwargs):
            return ''.join(map(chr, ids))
        def apply_chat_template(self, messages, tokenize, **kwargs):
            self.messages = messages
            rendered = 'CHAT' + messages[-1]['content']
            return self.encode(rendered) if tokenize else rendered
    tok = Tokenizer()
    ids, stats = chat_ids(tok, 'abcdefghij', 'Qwen/Qwen2.5-7B-Instruct', 'longbenchv2', 4)
    assert tok.decode(ids) == 'CHATabij'
    assert stats['truncated'] and stats['raw_prompt_tokens'] == 10
    assert tok.messages[0]['role'] == 'system'
    with pytest.raises(ValueError):
        chat_ids(tok, 'abcdefghij', 'qwen', 'longgenbench', 4)


def make_manifest(tmp_path, model='Qwen/Qwen2.5-72B-Instruct', bench='longbenchv2'):
    path = tmp_path / 'manifest.json'
    atomic_json(path, dict(model=model, revision='a'*40, benchmark=bench, smoke=False,
                           examples=[dict(id='one', token_ids=list(range(20)))], source_count=1))
    return path


def test_72b_shards_without_silently_substituting_models(tmp_path):
    path = make_manifest(tmp_path)
    plan = make_plan([path], tmp_path, 5, 140, ['full', 'freekv', 'rocketkv'], tmp_path)
    assert plan['jobs'][0]['layout']['gpus'] == 2
    assert plan['jobs'][1]['argv']
    assert plan['jobs'][2]['argv'] and plan['jobs'][2]['layout']['gpus'] == 2
    assert 'Qwen/device bridge' in plan['jobs'][2]['implementation']
    plan = make_plan([path], tmp_path, 1, 140, ['full'], tmp_path)
    assert plan['jobs'][0]['status'] == 'unsupported'


def test_report_does_not_publish_partial_or_smoke_as_full_score(tmp_path):
    path = make_manifest(tmp_path)
    plan = make_plan([path], tmp_path, 5, 140, ['full'], tmp_path)
    result_path = plan['jobs'][0]['output']
    result = dict(status='failed', expected=['one', 'two'], rows=[dict(id='one', accuracy=1.)])
    atomic_json(result_path, result)
    write_report(plan, tmp_path)
    rows = json.loads((tmp_path / 'comparison.json').read_text())
    assert rows[0]['accuracy'] is None
    result.update(status='completed', expected=['one'], smoke=True)
    atomic_json(result_path, result)
    write_report(plan, tmp_path)
    assert json.loads((tmp_path / 'comparison.json').read_text())[0]['accuracy'] is None
    result['smoke'] = False
    atomic_json(result_path, result)
    write_report(plan, tmp_path)
    assert json.loads((tmp_path / 'comparison.json').read_text())[0]['accuracy'] == 100


def test_scheduler_honors_visible_devices_and_continues_failures(tmp_path, monkeypatch):
    import threading
    import time
    from types import SimpleNamespace
    path = make_manifest(tmp_path)
    plan = make_plan([path], tmp_path, 5, 140, ['full', 'freekv'], tmp_path)
    mutex, active, observed = threading.Lock(), set(), []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,4,5,6,7')
    def fake_run(argv, env, **kwargs):
        devices = set(env['CUDA_VISIBLE_DEVICES'].split(','))
        with mutex:
            assert not active & devices
            active.update(devices)
            observed.append(devices)
        time.sleep(.02)
        with mutex:
            active.difference_update(devices)
        atomic_json(argv[argv.index('--out')+1], dict(status='completed', expected=['one'], rows=[dict(id='one', accuracy=0)]))
        return SimpleNamespace(returncode=1 if '--method' in argv and argv[argv.index('--method')+1] == 'full' else 0)
    monkeypatch.setattr('experiments.phase_one.subprocess.run', fake_run)
    assert not execute(plan, tmp_path)
    assert {frozenset(x) for x in observed} == {frozenset({'3', '4'}), frozenset({'5', '6'})}
    assert [j['status'] for j in plan['jobs']] == ['failed', 'completed']


def test_stale_judge_cannot_be_merged(tmp_path):
    path = make_manifest(tmp_path, bench='longgenbench')
    plan = make_plan([path], tmp_path, 5, 140, ['full'], tmp_path)
    output = Path(plan['jobs'][0]['output'])
    result = dict(status='completed', expected=['one'], smoke=False, rows=[dict(id='one', completion_rate=75)])
    atomic_json(output, result)
    atomic_json(output.with_name('judge.json'), dict(status='completed', generation_sha256='wrong', rows=[]))
    write_report(plan, tmp_path)
    row = json.loads((tmp_path/'comparison.json').read_text())[0]
    assert row['completion_rate'] == 75 and row['average_accuracy'] is None
