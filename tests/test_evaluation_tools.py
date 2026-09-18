import json
from types import SimpleNamespace
import pytest
import torch

from experiments.baselines import grouped_roundtrip, quantized_storage_bytes
from experiments.metrics import answer_scores, wilson, first_token_divergence
from experiments.prepare_longbench import prepare
from experiments.ablation_suite import variants, make_jobs
from experiments.summarize_results import paired_delta, summarize
from experiments import calibration as C
from kvtc import KVTCCodec,KVTCConfig


@pytest.mark.parametrize('bits',[(2,2),(4,2),(4,4)])
def test_grouped_baseline_tails_constants_residual_and_no_mutation(bits):
    torch.manual_seed(5)
    k=torch.randn(1,2,19,7); v=torch.randn_like(k)
    k[:,:,:4,0]=1e-9
    originals=(k.clone(),v.clone())
    layers,meta=grouped_roundtrip([(k,v)],*bits,group=4,residual=3)
    for src,orig,out in zip((k,v),originals,layers[0]):
        assert torch.equal(src,orig)
        assert torch.isfinite(out).all()
        assert torch.equal(out[:,:,-3:],src[:,:,-3:])
    assert not meta['packed_storage_implemented']
    # Only the altered key group changes; other key groups must not be affected.
    altered=k.clone(); altered[:,:,1,1]+=100
    other,_=grouped_roundtrip([(altered,v)],*bits,group=4,residual=3)
    assert torch.equal(layers[0][0][:,:,4:],other[0][0][:,:,4:])


def test_baseline_storage_includes_tail_metadata_and_residual():
    # 3 prefix tokens x 5 dims: K2=4 bytes, V4=8; K metadata=20,
    # V metadata=24; 2 residual tokens x 5 dims x K/V x 2 bytes=40.
    assert quantized_storage_bytes([(1,1,5,5)],2,4,group=4,residual=2)==96
    assert quantized_storage_bytes([(1,1,2,5)],2,4,group=4,residual=128)==40


def test_qa_metrics_multiple_references_and_interval_extremes():
    assert answer_scores('The BLUE river.',['red lake','blue river'])['exact_match']==1
    assert answer_scores('blue river flows',['blue river'])['token_f1']==pytest.approx(.8)
    assert answer_scores('wrong',['blue river'])['token_f1']==0
    assert wilson(40,40)[0]<1 and wilson(0,40)[1]>0
    with pytest.raises(ValueError): answer_scores('anything',[''])
    assert first_token_divergence(torch.tensor([1.,2.]),torch.tensor([1.,2.]))['first_token_kl']==0


class Tok:
    def __call__(self,text,**kwargs): return SimpleNamespace(input_ids=[ord(c) for c in text])


def test_full_dataset_no_question_leak_no_silent_truncation():
    rows=[dict(context='document '*100,input='secret question',answers=['answer'],_id='x')]
    with pytest.raises(ValueError,match='exceeds'): prepare(Tok(),rows,'model','qasper',600,[[1]],{})
    data=prepare(Tok(),rows,'model','qasper',600,[[1]],{},True)
    assert data['complete_task_split'] and len(data['documents'])==len(rows)
    doc=data['documents'][0]
    assert doc['context_truncated'] and data['truncated_documents']==1
    assert 'secret question' not in ''.join(map(chr,doc['token_ids']))
    assert doc['questions'][0]['fact_span'] is None
    assert len(doc['token_ids'])+len(doc['questions'][0]['token_ids'])+128==600


def test_plan_page_ablation_budget_and_revision_required(tmp_path):
    vs=dict(variants('ablation'))
    assert all(vs[n]['page']*vs[n]['recall_k']==1024 for n in ('base','page16','page64','page256'))
    manifest=tmp_path/'m.json'
    manifest.write_text(json.dumps(dict(model='Qwen/test',task='qasper',oracle_available=False)))
    with pytest.raises(ValueError,match='revision'): make_jobs([manifest],'comparison',tmp_path)
    data=json.loads(manifest.read_text()); data['revision']='pinned'; manifest.write_text(json.dumps(data))
    jobs=make_jobs([manifest],'comparison',tmp_path)
    assert len(jobs)==1 and 'random,scan' in jobs[0]['argv']
    with pytest.raises(ValueError,match='collision'): make_jobs([manifest,manifest],'comparison',tmp_path)


def test_paired_summary_refuses_missing_or_duplicate_controls():
    rows=[dict(document=i,question_index=0,style='q',arm=a,token_f1=s)
          for i in range(3) for a,s in [('vanilla',.8),('scan',.6)]]
    result=paired_delta(rows,'scan','vanilla',reps=100)
    assert result['mean_delta']==pytest.approx(-.2)
    assert result['document_count']==3
    with pytest.raises(ValueError,match='duplicate'): paired_delta(rows+rows[:1],'scan','vanilla')
    with pytest.raises(ValueError,match='missing'): paired_delta(rows[1:],'scan','vanilla')
    with pytest.raises(ValueError,match='incomplete'): summarize(dict(completed=False))


def test_calibration_checkpoint_identity_and_numerics(tmp_path):
    cfg=KVTCConfig(pca_rank_cap=4,block_sizes=(1,4),target_cr=2)
    codec=KVTCCodec(cfg,device='cpu')
    x=torch.randn(20,4); codec.calibrate([x],[x],verbose=False)
    path=tmp_path/'basis.pt'; C.save(codec,path,'test')
    other=KVTCCodec(cfg,device='cpu'); C.load(other,path,'test')
    assert torch.equal(other.art.key.V,codec.art.key.V)
    assert other.art.assignments==codec.art.assignments
    with pytest.raises(ValueError,match='identity'): C.load(other,path,'wrong')
