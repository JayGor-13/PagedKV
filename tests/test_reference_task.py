import random
from types import SimpleNamespace

import torch

from experiments import reference_task as D
from experiments.run_rare_facts import parser, prepare_manifest


class CharacterTokenizer:
    """Small deterministic tokenizer for generator tests, not model evaluation."""
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(c) for c in text])

    def decode(self, ids):
        return ''.join(chr(int(i)) for i in ids)


def test_generator_reproducible_facts_survive_and_question_styles_differ():
    tok = CharacterTokenizer()
    chunks = [(f'Filler passage {i}. ' * 30)[:300] for i in range(80)]
    for make, length in [(D.make_doc, 1800), (D.make_long_doc, 4096)]:
        a = make(random.Random(3), tok, chunks, target_len=length)
        b = make(random.Random(3), tok, chunks, target_len=length)
        assert torch.equal(a[0], b[0])
        assert a[1:] == b[1:]
        for name, code in zip(a[1], a[2]):
            start, end = D.fact_span(a[0], tok, code)
            assert start >= 0 and tok.decode(a[0][start:end]) == code
            assert code not in tok.decode(D.question_for(tok, name))
            assert not torch.equal(D.question_for(tok, name), D.paraphrase_question(tok, name))


def test_frozen_manifest_reused_without_regenerating_data(tmp_path, monkeypatch):
    tok = CharacterTokenizer()
    chunks = [(f'Filler passage {i}. ' * 30)[:300] for i in range(80)]
    monkeypatch.setattr(D, 'build_filler_chunks', lambda t: chunks)
    monkeypatch.setattr(D, 'load_corpus', lambda t, n, ctx: [torch.arange(ctx)]*n)
    args = parser().parse_args(['--manifest', str(tmp_path/'data.json'), '--docs', '2'])
    first = prepare_manifest(tok, args)
    monkeypatch.setattr(D, 'build_filler_chunks', lambda t: (_ for _ in ()).throw(AssertionError('regenerated')))
    second = prepare_manifest(tok, args)
    assert first == second
