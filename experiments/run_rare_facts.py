"""Run the reference rare-fact task through OUR indexed KVTC archive.

Quality/control harness, NOT a GPU latency or equal-memory benchmark. All
selector scans and full-decode controls are explicitly separated from recall.
"""
import argparse
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import platform
import random
import sys
from dataclasses import asdict

import torch

from kvtc import KVTCCodec, KVTCConfig, feature_dim, restore_kv_for_model
from kvtc.cold_store import ColdStore
from . import model_adapter as A
from . import reference_task as D
from .reference_heuristics import eviction_scores, eviction_mask, page_mass_scores, pages_to_mask
from .selector import scan_key_coefficients
from . import calibration as C
from .baselines import grouped_roundtrip
from .metrics import answer_scores, measured, first_token_divergence, wilson
from .benchmark_scores import score_question


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_manifest(tok, args):
    """Freeze calibration/document/question token IDs before any model inference."""
    path = Path(args.manifest)
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['model'] != args.model:
            raise ValueError('manifest model/tokenizer ID differs from --model')
        if data.get('revision') and data['revision'] != args.revision:
            raise ValueError('manifest revision differs from --revision')
        if data.get('tokenizer_sha256') and hasattr(tok, 'backend_tokenizer'):
            actual = hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest()
            if actual != data['tokenizer_sha256']:
                raise ValueError('manifest tokenizer content differs')
        return data
    rng = random.Random(args.seed)
    chunks = D.build_long_chunks(tok) if args.protocol == 'long' else D.build_filler_chunks(tok)
    documents = []
    for i in range(args.docs):
        if args.protocol == 'long':
            ids, names, codes, _ = D.make_long_doc(rng, tok, chunks, args.ctx)
        else:
            ids, names, codes, _ = D.make_doc(rng, tok, chunks, target_len=args.ctx)
        questions = []
        styles = [('exact', i % len(names))]
        if args.protocol == 'long':
            styles.append(('para', (i+1) % len(names)))
        for style, j in styles:
            q = D.question_for(tok, names[j]) if style == 'exact' else D.paraphrase_question(tok, names[j])
            span = D.fact_span(ids, tok, codes[j])
            if span[0] < 0:
                raise ValueError('target fact missing from tokenized document')
            questions.append(dict(style=style, token_ids=q.tolist(), target=codes[j],
                                  target_name=names[j], fact_span=list(span)))
        documents.append(dict(id=i, token_ids=ids.tolist(), questions=questions))
    data = dict(schema=1, model=args.model, revision=args.revision, protocol=args.protocol, ctx=args.ctx, seed=args.seed,
                python=platform.python_version(), filler_chunks_sha256=digest(chunks),
                calibration_evaluation_text_overlap=True,
                historical_reference_token_identity='unknown; historical token IDs were not supplied',
                calibration=[d.tolist() for d in D.load_corpus(tok, args.calib_docs, args.calib_ctx)],
                documents=documents)
    if hasattr(tok, 'backend_tokenizer'):
        data['tokenizer_sha256'] = hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1)+'\n', encoding='utf-8')
    return data


def save_report(path, report):
    grouped = defaultdict(list)
    for row in report['answers']:
        grouped[row['style']+'/'+row['arm']].append(row['contains_target'])
    report['accuracy'] = {name: dict(correct=sum(xs), total=len(xs), rate=sum(xs)/len(xs),
                                   wilson95=wilson(sum(xs),len(xs)))
                          for name, xs in grouped.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--revision', default=None)
    p.add_argument('--protocol', choices=('short', 'long'), default='short')
    p.add_argument('--ctx', type=int, default=1800)
    p.add_argument('--docs', type=int, default=4)
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--calib-docs', type=int, default=12)
    p.add_argument('--calib-ctx', type=int, default=2048)
    p.add_argument('--cold-cr', type=float, default=16.)
    p.add_argument('--rank-cap', type=int, default=8192)
    p.add_argument('--dp-stride', type=int, default=16,
                   help='16 matches the reference search restriction; 1 is unrestricted')
    p.add_argument('--page', type=int, default=128)
    p.add_argument('--hot-tokens', type=int, default=256,
                   help='Explicit hot budget; NOT a claimed byte-matched reference hot-CR')
    p.add_argument('--recall-k', type=int, default=8)
    p.add_argument('--halo', type=int, default=1)
    p.add_argument('--topk', type=int, default=256)
    p.add_argument('--band', default='14,21')
    p.add_argument('--selection', default='oracle,random,scan')
    p.add_argument('--new-tokens', type=int, default=10)
    p.add_argument('--prefill-chunk', type=int, default=2048)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--gpus', type=int, default=1, help='Model shards; >1 uses GPU-only balanced placement')
    p.add_argument('--gpu-memory-gib', type=int, default=120)
    p.add_argument('--dtype', choices=('bfloat16','float16'), default='bfloat16')
    p.add_argument('--manifest', default='outputs/rare_facts_manifest.json')
    p.add_argument('--out', default='outputs/rare_facts_results.json')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--baselines', default='h2o,snap,recent,kivi4,kivi2,int4',
                   help='Local quality controls; comma-separated, or none')
    p.add_argument('--refine', type=int, default=0, help='Extra scan rounds using reconstructed active entries')
    p.add_argument('--quant-group', type=int, default=32)
    p.add_argument('--residual', type=int, default=128)
    p.add_argument('--calibration-dir', help='Reuse identity-checked artifacts across ablations')
    p.add_argument('--resume', action='store_true', help='Resume after fully completed documents')
    p.add_argument('--overwrite', action='store_true', help='Explicitly replace an existing report')
    p.add_argument('--svd-method', choices=('gram_eigh','randomized'), default='gram_eigh')
    p.add_argument('--stop-eos', action='store_true', help='Stop on model EOS; fixed length remains default for rare facts')
    return p


@torch.no_grad()
def main():
    args = parser().parse_args()
    if min(args.docs, args.ctx, args.calib_docs, args.calib_ctx, args.page, args.new_tokens,
           args.prefill_chunk, args.recall_k, args.topk, args.quant_group, args.dp_stride,
           args.rank_cap) < 1 or args.hot_tokens < 132 or min(args.halo,args.refine,args.residual) < 0:
        raise ValueError('positive sizes required; hot-tokens must be at least 132')
    selections = args.selection.split(',')
    if set(selections)-{'oracle', 'random', 'scan'} or len(set(selections)) != len(selections):
        raise ValueError('unknown selection method')
    baselines = [] if args.baselines == 'none' else args.baselines.split(',')
    if set(baselines)-{'h2o','snap','recent','kivi4','kivi2','int4'} or len(set(baselines)) != len(baselines):
        raise ValueError('unknown or duplicate baseline')
    if args.refine and 'scan' not in selections:
        raise ValueError('--refine requires scan selection')
    if args.resume and args.overwrite:
        raise ValueError('--resume and --overwrite are mutually exclusive')
    output = Path(args.out)
    if output.exists() and not args.prepare_only and not (args.resume or args.overwrite):
        raise FileExistsError('output exists: use --resume, --overwrite or a new path')
    if not args.prepare_only and args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Run on the H200; CPU tests do not reproduce model accuracy.')
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    data = prepare_manifest(tok, args)
    if args.prepare_only:
        print(f"Frozen {len(data['documents'])} documents in {args.manifest}; no model weights loaded.")
        return
    if not data['calibration'] or not data['documents']:
        raise ValueError('manifest must contain calibration and evaluation data')
    if len({d['id'] for d in data['documents']}) != len(data['documents']):
        raise ValueError('duplicate document IDs in manifest')
    if any(not d['token_ids'] or not d['questions'] or any(not q['token_ids'] for q in d['questions']) for d in data['documents']):
        raise ValueError('empty document or question in manifest')
    if 'oracle' in selections and any(q.get('fact_span') is None for d in data['documents'] for q in d['questions']):
        raise ValueError('oracle requires annotated spans; use --selection random,scan for LongBench')
    # Validate/skip a finished pinned run before loading model weights or calibration.
    if args.resume and output.exists() and args.revision:
        previous = json.loads(output.read_text(encoding='utf-8'))
        files = sorted([*Path('kvtc').glob('*.py'), *Path('experiments').glob('*.py')])
        hashes = {p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        expected_identity = digest([data, {k:v for k,v in vars(args).items() if k not in ('resume','overwrite','out','calibration_dir')},hashes,args.revision])
        if previous.get('run_identity') != expected_identity:
            raise ValueError('resume identity mismatch: config, manifest, source or model changed')
        if previous.get('completed'):
            if set(previous['completed_documents']) != {d['id'] for d in data['documents']}:
                raise ValueError('completed report has missing documents')
            print(f'Already complete: {output}', flush=True)
            return
    torch.manual_seed(args.seed)
    from .gpu_layout import model_load_kwargs, assert_gpu_only
    placement = model_load_kwargs(args.gpus, args.gpu_memory_gib, args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision,
        dtype=getattr(torch,args.dtype) if args.device == 'cuda' else torch.float32,
        attn_implementation='sdpa', **placement).eval()
    if not placement:
        model.to(args.device)
    assert_gpu_only(model)
    A.check_model(model)
    band = (0,model.config.num_hidden_layers) if args.band == 'all' else tuple(map(int, args.band.split(',')))
    if len(band) != 2 or not 0 <= band[0] < band[1] <= model.config.num_hidden_layers:
        raise ValueError('invalid layer band')
    rank = min(args.rank_cap, feature_dim(model))
    if rank % args.dp_stride:
        raise ValueError('rank must be divisible by dp-stride; use 1')
    codec = KVTCCodec(KVTCConfig(target_cr=args.cold_cr, pca_rank_cap=rank,
                                 svd_method=args.svd_method, seed=args.seed,
                                 dp_stride=args.dp_stride), device=args.device)
    source_files = sorted([*Path('kvtc').glob('*.py'), *Path('experiments').glob('*.py')])
    source_hashes = {p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    revision = getattr(model.config, '_commit_hash', None) or args.revision
    signature = C.identity(args.model, revision, codec.cfg, data['calibration'],data.get('tokenizer_sha256'))
    signature = digest([signature,{k:v for k,v in source_hashes.items() if k.startswith('kvtc')}])
    checkpoint = Path(args.calibration_dir)/f'{signature}.pt' if args.calibration_dir else None
    def calibrate():
        if checkpoint and checkpoint.exists():
            C.load(codec,checkpoint,signature)
            return 'reused'
        keys, values = [], []
        for ids in data['calibration']:
            layers, _ = A.prefill(model, torch.tensor(ids), args.prefill_chunk)
            k, v = A.to_features(model, layers)
            keys.append(k); values.append(v)
            del layers
        codec.calibrate(keys, values, verbose=True)
        if checkpoint:
            C.save(codec,checkpoint,signature)
        return 'fitted'
    print(f'Calibration feature dimension={feature_dim(model)}, rank={rank}, method={args.svd_method}',flush=True)
    calibration_status, calibration_ms = measured(calibrate,args.device)
    run_identity = digest([data, {k:v for k,v in vars(args).items() if k not in ('resume','overwrite','out','calibration_dir')},source_hashes,revision])
    report = dict(config=vars(args), manifest_sha256=digest(data), model_revision=getattr(model.config, '_commit_hash', None),
                  schema=2, run_identity=run_identity, source_hashes=source_hashes,
                  expected_documents=len(data['documents']), completed_documents=[],
                  dataset_protocol=data.get('protocol'), source=data.get('source'),
                  dataset_metadata={k:data.get(k) for k in ('complete_task_split','truncation','truncated_documents',
                      'prompt_protocol','calibration_evaluation_text_overlap')},
                  environment=dict(python=sys.version, torch=str(torch.__version__),
                      transformers=__import__('transformers').__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None),
                  calibration=dict(signature=signature,status=calibration_status,ms=calibration_ms,config=asdict(codec.cfg)),
                  run_kind='quality_and_integration_controls', completed=False,
                  warning='CPU zlib archive; scan selector reads every key stream; not GPU-only or byte-matched',
                  baseline_limitations='global approximate H2O/Snap; local grouped quantization, no packed low-bit kernels; no total-byte matching',
                  shared_calibration_resident_bytes=sum(t.numel()*t.element_size()
                      for b in (codec.art.key, codec.art.value) for t in (b.mu, b.V, b.evals)),
                  documents=[], answers=[])
    if args.resume and output.exists():
        previous = json.loads(output.read_text(encoding='utf-8'))
        if previous.get('run_identity') != run_identity:
            raise ValueError('resume identity mismatch: config, manifest, source or model changed')
        report = previous
        done = set(report['completed_documents'])
        report['answers'] = [r for r in report['answers'] if r['document'] in done]
        report['documents'] = [r for r in report['documents'] if r['id'] in done]
    done = set(report['completed_documents'])
    eos = getattr(model.generation_config,'eos_token_id',None)
    eos_ids = ([] if eos is None else eos if isinstance(eos,list) else [eos]) if args.stop_eos else []
    save_report(output,report)
    for doc in data['documents']:
        if doc['id'] in done:
            continue
        ids = torch.tensor(doc['token_ids'])
        n = len(ids)
        if any(n+len(q['token_ids'])+q.get('max_new_tokens',args.new_tokens) > model.config.max_position_embeddings
               for q in doc['questions']):
            raise ValueError(f"document {doc['id']} exceeds model context capacity; no silent truncation")
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        (layers, queries), prefill_ms = measured(lambda: A.prefill(model, ids, args.prefill_chunk, collect_queries=True),args.device)
        k, v = A.to_features(model, layers)
        (h2o, snap), hot_scoring_ms = measured(lambda: eviction_scores(model, layers, queries, n, args.device, stride=16, win=min(64, n)),args.device)
        del queries
        hot_mask = eviction_mask(h2o, n, min(n, args.hot_tokens))
        hot_mask[:4] = 1; hot_mask[max(0, n-128):] = 1
        # Preserve protected entries even when the reference half-window split is smaller.
        hot_ids = hot_mask.nonzero().flatten()
        hot_layers = A.slice_layers(layers, hot_ids)
        archive, compress_ms = measured(
            lambda: ColdStore.encode(codec, k, v, args.page, key_head_rank=args.topk), args.device)
        full, full_decode_ms = measured(archive.decode_all,args.device)  # Diagnostic only.
        full_layers = restore_kv_for_model(full.keys, full.values, model, device=args.device)
        monolithic = codec.compress(k, v)
        mono_k, mono_v = codec.decompress(monolithic)
        mono_layers = restore_kv_for_model(mono_k, mono_v, model, device=args.device)
        (coefficients, scan_stats), scan_ms = (measured(lambda: scan_key_coefficients(archive,args.topk),args.device)
                                    if 'scan' in selections else ((None, {}),0.))
        report['documents'].append(dict(id=doc['id'], tokens=n, archive_bytes=archive.nbytes(),
                                        source_id=doc.get('source_id'),context_truncated=doc.get('context_truncated',False),
                                        original_document_tokens=doc.get('original_document_tokens',n),
                                        page_count=archive.page_count,index_bytes=archive.index_bytes,
                                        raw_feature_fp16_bytes=n*feature_dim(model)*4,
                                        actual_archive_cr=(n*feature_dim(model)*4)/archive.nbytes(),
                                        prefill_ms=prefill_ms,hot_scoring_ms=hot_scoring_ms,
                                        compression_ms=compress_ms,full_decode_ms=full_decode_ms,key_scan_ms=scan_ms,
                                        monolithic_bytes=sum(p.nbytes() for p in monolithic.values()),
                                        hot_tokens=len(hot_ids), hot_bytes=A.tensor_bytes(hot_layers),
                                        scan=scan_stats))
        del k, v, full, mono_k, mono_v, monolithic
        for qi,question in enumerate(doc['questions']):
            qids = torch.tensor(question['token_ids'])
            target = question['target']
            reference_logits = None
            answers = question.get('answers',[target])

            def evaluate(arm, active_layers, **extra):
                nonlocal reference_logits
                generated, logits, timing = A.answer_profile(model,active_layers,n,qids,
                    question.get('max_new_tokens',args.new_tokens),eos_ids=eos_ids)
                text = tok.decode(generated,skip_special_tokens=True)
                if arm == 'vanilla':
                    reference_logits = logits
                scores = score_question(text,question)
                scores.update(first_token_divergence(reference_logits,logits))
                active_count = active_layers[0][0].shape[2]
                report['answers'].append(dict(document=doc['id'], question_index=qi, style=question['style'], arm=arm,
                    target=target, generated_ids=generated, generated_text=text,
                    references=answers, **scores, timing=timing,
                    active_tokens=active_count,attended_fraction=active_count/n,
                    active_cache_bytes=A.tensor_bytes(active_layers), **extra))

            evaluate('vanilla', layers)
            evaluate('monolithic_kvtc_full', mono_layers)
            evaluate('paged_kvtc_full', full_layers)
            evaluate('hot_only', hot_layers)
            for baseline in baselines:
                if baseline in ('h2o','snap','recent'):
                    score = {'h2o':h2o,'snap':snap,'recent':torch.arange(n,dtype=torch.float32)}[baseline]
                    selected = eviction_mask(score,n,min(n,len(hot_ids))).nonzero().flatten()
                    baseline_layers = A.slice_layers(layers,selected)
                    evaluate(baseline+'_approx',baseline_layers,positions=selected.tolist())
                else:
                    kb,vb = {'kivi4':(4,2),'kivi2':(2,2),'int4':(4,4)}[baseline]
                    (baseline_layers,quant_stats),quant_ms = measured(lambda: grouped_roundtrip(
                        layers,kb,vb,args.quant_group,args.residual),args.device)
                    evaluate(baseline+'_local_full',baseline_layers,quantization_ms=quant_ms,**quant_stats)
                del baseline_layers
            span = question.get('fact_span')
            for selection in selections:
                prepass_ms = score_ms = 0.
                if selection == 'oracle':
                    st,en = span
                    lo, hi = D.sentence_span(st, en, n)
                    pages = list(range(lo//args.page, (hi-1)//args.page+1))
                elif selection == 'random':
                    rng = random.Random(args.seed+doc['id'])
                    pages = rng.sample(range(archive.page_count), min(args.recall_k, archive.page_count))
                else:
                    (prepass, qs),prepass_ms = measured(lambda: A.question_forward(model,hot_layers,n,qids,collect_queries=True),args.device)
                    del prepass
                    scores,score_ms = measured(lambda: page_mass_scores(model, codec, coefficients, n, qs, args.page,
                                              args.topk, hot_mask, args.device, band),args.device)
                    pages = scores.argsort(descending=True)[:args.recall_k].tolist()
                    del qs
                mask = pages_to_mask(pages, n, args.page, args.halo)
                page_ids = sorted(set((mask.nonzero().flatten()//args.page).tolist()))
                active,recovery_ms = measured(lambda: A.recover(model,archive,page_ids,hot_layers,hot_ids),args.device)
                cumulative_bytes=active.payload_bytes_read
                for iteration in range(1+(args.refine if selection == 'scan' else 0)):
                    if iteration:
                        (prepass,qs),ms = measured(lambda: A.question_forward(model,active.layers,n,qids,collect_queries=True),args.device)
                        del prepass
                        prepass_ms += ms
                        scores,ms = measured(lambda: page_mass_scores(model,codec,coefficients,n,qs,args.page,
                                               args.topk,hot_mask,args.device,band),args.device)
                        score_ms += ms
                        del qs
                        pages=scores.argsort(descending=True)[:args.recall_k].tolist()
                        mask=pages_to_mask(pages,n,args.page,args.halo)
                        page_ids=sorted(set((mask.nonzero().flatten()//args.page).tolist()))
                        del active
                        active,ms=measured(lambda: A.recover(model,archive,page_ids,hot_layers,hot_ids),args.device)
                        recovery_ms += ms
                        cumulative_bytes += active.payload_bytes_read
                    evaluate(selection if not iteration else f'scan_refine{iteration}',active.layers,pages=page_ids,
                        selected_payload_bytes=active.payload_bytes_read,cumulative_recovery_payload_bytes=cumulative_bytes,
                        pages_decoded=len(page_ids),decoded_page_fraction=len(page_ids)/archive.page_count,
                        selected_payload_fraction=active.payload_bytes_read/(archive.nbytes()-archive.index_bytes),
                        positions=active.positions.tolist(),
                        target_span_covered=bool(torch.isin(torch.arange(*span),active.positions).all()) if span else None,
                        extra_query_prepasses=1+iteration if selection=='scan' else 0,
                        retrieval_prepass_ms=prepass_ms,page_scoring_ms=score_ms,recovery_ms=recovery_ms,
                        retrieval_ms=prepass_ms+score_ms+recovery_ms,
                        recall_timing_excludes_once_per_document_key_scan=True)
                del active
            save_report(output, report)
        report['documents'][-1]['quality_harness_peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated() if args.device=='cuda' else None
        report['documents'][-1]['quality_harness_peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved() if args.device=='cuda' else None
        report['completed_documents'].append(doc['id'])
        save_report(output, report)
        del layers, hot_layers, full_layers, mono_layers, archive, coefficients
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        print(f"Finished document {doc['id']}; results: {output}", flush=True)
    report['completed'] = True
    save_report(output, report)


if __name__ == '__main__':
    main()
