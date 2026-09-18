"""Offline end-to-end wiring control using a tiny RANDOM-WEIGHT Qwen2 model.

No pretrained weights/downloads. Accuracy from this fixture is meaningless.
"""
import json
from pathlib import Path
import sys
import tempfile

import torch


def main():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from experiments.run_rare_facts import main as run

    torch.manual_seed(71)
    torch.set_num_threads(2)
    Path('work').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tiny_adapter_', dir='work') as folder:
        root = Path(folder).resolve()
        tokenizer = Tokenizer(WordLevel({**{'[UNK]': 0}, **{f'w{i}': i for i in range(1, 128)}}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]').save_pretrained(root)
        cfg = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
        Qwen2ForCausalLM(cfg).save_pretrained(root)
        manifest = dict(schema=1, model=str(root), fixture='random_weight_wiring_only',
                        calibration=[torch.randint(1, 128, (96,)).tolist() for _ in range(2)],
                        documents=[dict(id=0, token_ids=torch.randint(1, 128, (180,)).tolist(),
                            questions=[dict(style='exact', token_ids=[11, 12, 13], target='482913',
                                            fact_span=[70, 72], target_name='fixture')])])
        (root/'manifest.json').write_text(json.dumps(manifest))
        output = Path('outputs/model_integration_smoke.json').resolve()
        old_argv = sys.argv
        try:
            sys.argv = ['run_rare_facts', '--model', str(root), '--manifest', str(root/'manifest.json'),
                        '--device', 'cpu', '--out', str(output), '--band', '0,2', '--page', '16',
                        '--topk', '16', '--rank-cap', '32', '--dp-stride', '1', '--cold-cr', '2',
                        '--hot-tokens', '132', '--recall-k', '1', '--halo', '0', '--new-tokens', '2',
                        '--refine','1','--overwrite','--calibration-dir',str(root/'calibration')]
            run()
            # Resume must load the same calibration artifact and not duplicate rows.
            first=json.loads(output.read_text())
            sys.argv[sys.argv.index('--overwrite')]='--resume'
            run()
            second=json.loads(output.read_text())
            assert first['answers']==second['answers']
            assert first['completed_documents']==second['completed_documents']
        finally:
            sys.argv = old_argv
        result = json.loads(output.read_text())
        result['run_kind'] = 'random_weight_model_wiring_only'
        result['accuracy_is_meaningful'] = False
        expected = {'vanilla', 'monolithic_kvtc_full', 'paged_kvtc_full', 'hot_only', 'oracle', 'random', 'scan',
                    'h2o_approx','snap_approx','recent_approx','kivi4_local_full','kivi2_local_full','int4_local_full','scan_refine1'}
        assert {row['arm'] for row in result['answers']} == expected
        assert result['completed']
        assert result['completed_documents']==[0]
        refined=next(r for r in result['answers'] if r['arm']=='scan_refine1')
        assert refined['extra_query_prepasses']==2
        assert refined['cumulative_recovery_payload_bytes']>=refined['selected_payload_bytes']
        assert all(r['timing']['generation_ms']>=0 for r in result['answers'])
        output.write_text(json.dumps(result, indent=2)+'\n')
        print('All 14 arms completed on the tiny random model. No accuracy claim.')
        print(output)


if __name__ == '__main__':
    main()
