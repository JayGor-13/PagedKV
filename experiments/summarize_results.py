"""Export raw metrics and document-paired bootstrap intervals from one run.

Never silently mix datasets, seeds or partial runs. Intervals describe this dataset
sample; document clustering keeps repeated questions from being independent units.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics


def paired_delta(rows,arm,reference,metric='token_f1',reps=2000,seed=0):
    indexed={}
    for r in rows:
        key=(r['document'],r.get('question_index',r['style']),r['arm'])
        if key in indexed:
            raise ValueError('duplicate paired observation')
        indexed[key]=r
    clusters=defaultdict(list)
    for (doc,q,a),row in indexed.items():
        if a!=arm:
            continue
        other=indexed.get((doc,q,reference))
        if other is None:
            raise ValueError('missing paired reference row')
        clusters[doc].append(float(row[metric])-float(other[metric]))
    if not clusters:
        return None
    values=[sum(v)/len(v) for v in clusters.values()]
    rng=random.Random(seed)
    samples=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(reps))
    return dict(document_count=len(values),mean_delta=sum(values)/len(values),
        bootstrap95=[samples[int(.025*reps)],samples[min(reps-1,int(.975*reps))]],
        method='paired_document_bootstrap',reps=reps,seed=seed)


def summarize(report,allow_partial=False):
    complete=report.get('completed') and len(report.get('completed_documents',[]))==report.get('expected_documents')
    if not complete and not allow_partial:
        raise ValueError('incomplete report; --allow-partial is required and output remains labeled partial')
    grouped=defaultdict(list)
    for r in report['answers']:
        grouped[(r['style'],r['arm'])].append(r)
    result=dict(completed=bool(complete),manifest_sha256=report['manifest_sha256'],groups={})
    for (style,arm),rows in grouped.items():
        scores={metric:sum(float(r[metric]) for r in rows)/len(rows)
                for metric in ('token_f1','exact_match','contains_target','attended_fraction','first_token_kl')}
        generation=[r['timing']['generation_ms'] for r in rows]
        retrieval=[r.get('retrieval_ms',0.) for r in rows]
        scores.update(n=len(rows),median_generation_ms=statistics.median(generation),
                      median_retrieval_plus_generation_ms=statistics.median([a+b for a,b in zip(generation,retrieval)]),
                      excludes_once_per_document_scan=True,
                      quality_harness_timing_only=True,
                      paired_f1_vs_vanilla=paired_delta([r for r in report['answers'] if r['style']==style],arm,'vanilla'))
        result['groups'][style+'/'+arm]=scores
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report')
    p.add_argument('--out-dir',required=True)
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    report=json.loads(Path(args.report).read_text(encoding='utf-8'))
    summary=summarize(report,args.allow_partial)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    rows=[]
    for row in report['answers']:
        flat={k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()}
        rows.append(flat)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'per_example.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    docs=[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in d.items()} for d in report['documents']]
    fields=sorted(set().union(*(d.keys() for d in docs)))
    with (out/'per_document.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(docs)
    print(out/'summary.json')


if __name__=='__main__':
    main()
