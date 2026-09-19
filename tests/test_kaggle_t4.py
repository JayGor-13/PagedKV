import json
from argparse import Namespace

import pytest
import torch

from experiments.t4_attention import sdpa_attention
from experiments.kaggle_t4 import PROFILE, method_config, protocol, select_rows, write_report, result_path
from experiments.benchmark_state import atomic_json, digest


@pytest.mark.parametrize('nq,nk,causal', [(9, 9, True), (1, 9, True), (3, 9, True), (9, 3, True), (3, 9, False)])
@pytest.mark.parametrize('kv_heads', [1, 4])
def test_t4_sdpa_matches_explicit_bottom_right_attention(nq, nk, causal, kv_heads):
    torch.manual_seed(7)
    q = torch.randn(1, nq, 4, 8)
    k, v = torch.randn(1, nk, kv_heads, 8), torch.randn(1, nk, kv_heads, 8)
    keys, values = k.repeat_interleave(4 // kv_heads, 2), v.repeat_interleave(4 // kv_heads, 2)
    logits = q.transpose(1, 2) @ keys.transpose(1, 2).transpose(-1, -2) * .25
    if causal:
        mask = torch.arange(nk)[None, :] <= torch.arange(nq)[:, None] + nk - nq
        logits.masked_fill_(~mask, -torch.inf)
    expected = (torch.nan_to_num(logits.softmax(-1)) @ values.transpose(1, 2)).transpose(1, 2)
    actual = sdpa_attention(q, k, v, causal=causal, softmax_scale=.25)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_t4_rejects_unsupported_attention_options():
    x = torch.randn(1, 3, 2, 8)
    with pytest.raises(NotImplementedError):
        sdpa_attention(x, x, x, window_size=(2, 2))
    with pytest.raises(ValueError):
        sdpa_attention(x.expand(2, -1, -1, -1), x, x)


def test_t4_profile_limits_and_smoke():
    args = Namespace(samples=4, prompt_cap=2048, max_new_tokens=128, smoke=True,
                     gpus=1, selection='stratified', context_capacity_pct=None,
                     models=['Qwen/Qwen2.5-0.5B-Instruct'],
                     methods=['ours', 'full'], benchmarks=['longbenchv2'])
    cfg = protocol(args)
    assert cfg['samples'] == 1 and cfg['max_new_tokens'] == 8 and cfg['codec_rank'] == 64
    args.prompt_cap = 120000
    with pytest.raises(ValueError):
        protocol(args)


def test_t4_large_models_require_dual_gpu_profile():
    args = Namespace(samples=100, prompt_cap=4096, max_new_tokens=128, smoke=False, gpus=1,
                     selection='stratified', context_capacity_pct=12.5,
                     models=['Qwen/Qwen2.5-7B-Instruct'], methods=['full'], benchmarks=['longbenchv2'])
    with pytest.raises(ValueError, match='require --gpus 2'):
        protocol(args)
    args.gpus = 2
    cfg = protocol(args)
    assert cfg['profile'] == 'kaggle-t4-dual-v1' and cfg['gpus'] == 2 and cfg['samples'] == 100


def test_t4_twelve_point_five_percent_capacity_matches_benchmark_budget():
    cfg = dict(prompt_cap=16384, max_new_tokens=128, page_size=32,
               sink=64, recent=64, budget=256, context_capacity_pct=12.5)
    longbench = method_config(cfg, 'longbenchv2', 'quest')
    longgen = method_config(cfg, 'longgenbench', 'quest')
    for resolved in (longbench, longgen):
        assert resolved['context_capacity_target_tokens'] == 2048
        assert resolved['sink'] + resolved['recent'] + resolved['budget'] == 2048


def test_t4_stratified_selection_is_stable_and_covers_groups():
    rows = [dict(id=f'{group}-{i}', domain=group, difficulty='easy', length='short')
            for group in ('a', 'b', 'c') for i in range(5)]
    first = select_rows(rows, 6, 'longbenchv2', 'stratified')
    second = select_rows(rows, 6, 'longbenchv2', 'stratified')
    assert first == second and {row['domain'] for row in first} == {'a', 'b', 'c'}


def test_t4_report_suppresses_smoke_failed_and_mismatched_results(tmp_path):
    manifest = dict(model='Qwen/Qwen2.5-0.5B-Instruct', benchmark='longbenchv2', smoke=False, examples=[dict(id='a')])
    cfg = dict(methods=['full'], protocol_note='Shortened experiment')
    frozen = dict(config=cfg, manifests=[manifest])
    path = result_path(tmp_path, manifest, 'full')
    state = dict(profile=PROFILE, status='completed', manifest_sha256=digest(manifest),
                 rows=[dict(id='a', accuracy=1., output_tokens=3, elapsed_seconds=1.)])
    def score():
        write_report(tmp_path, frozen)
        return json.loads((tmp_path / 'comparison.json').read_text())['rows'][0]['subset_accuracy_pct']
    atomic_json(path, state)
    assert score() == 100
    atomic_json(tmp_path / 'worker_failures.json', {str(path.relative_to(tmp_path)): 'identity mismatch'})
    assert score() is None
    atomic_json(tmp_path / 'worker_failures.json', {})
    state['manifest_sha256'] = 'wrong'
    atomic_json(path, state)
    assert score() is None
    manifest['smoke'] = True
    state['manifest_sha256'] = digest(manifest)
    atomic_json(path, state)
    assert score() is None


def test_t4_report_does_not_rescale_upstream_completion_percentage(tmp_path):
    manifest = dict(model='Qwen/Qwen2.5-0.5B-Instruct', benchmark='longgenbench', smoke=False,
                    examples=[dict(id='a'), dict(id='b')])
    frozen = dict(config=dict(methods=['full'], protocol_note='Shortened experiment'), manifests=[manifest])
    path = result_path(tmp_path, manifest, 'full')
    atomic_json(path, dict(profile=PROFILE, status='completed', manifest_sha256=digest(manifest),
        rows=[dict(id='a', completion_rate=25., output_tokens=10, elapsed_seconds=2.),
              dict(id='b', completion_rate=75., output_tokens=20, elapsed_seconds=4.)]))
    write_report(tmp_path, frozen)
    row = json.loads((tmp_path / 'comparison.json').read_text())['rows'][0]
    assert row['shortened_completion_pct'] == 50.
    assert row['elapsed_sample_seconds'] == 3.
    assert row['elapsed_total_seconds'] == 6.
    assert row['effective_output_tokens_per_second'] == 5.
