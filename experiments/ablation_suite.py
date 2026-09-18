"""Create explicit comparison/one-factor ablation plans; optionally execute locally.

Run only on the chosen GPU host. Each job is a separate subprocess. Completed
results are reused only through the runner's manifest/config/source identity check.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def variants(profile):
    base=dict(page=128,hot_tokens=1024,recall_k=8,halo=1,topk=256,cold_cr=16,
              refine=0,band='14,21')
    result=[('base',base)]
    if profile == 'ablation':
        changes=[('page16',dict(page=16,recall_k=64)),('page32',dict(page=32,recall_k=32)),('page64',dict(page=64,recall_k=16)),
                 ('page256',dict(page=256,recall_k=4)),('coords64',dict(topk=64)),
                 ('coords128',dict(topk=128)),('coords512',dict(topk=512)),
                 ('hot512',dict(hot_tokens=512)),('hot2048',dict(hot_tokens=2048)),
                 ('recall4',dict(recall_k=4)),('recall16',dict(recall_k=16)),
                 ('no_halo',dict(halo=0)),('refine1',dict(refine=1)),
                 ('cr8',dict(cold_cr=8)),('cr32',dict(cold_cr=32)),
                 ('all_layers',dict(band='all'))]
        result += [(name,{**base,**delta}) for name,delta in changes]
    return result


def make_jobs(manifests, profile, out_dir, seeds=(3,)):
    jobs=[]
    for path in manifests:
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if not data.get('revision'):
            raise ValueError(f'{path}: pin a model revision in the manifest first')
        task=data.get('task',data.get('protocol','rare_facts'))
        model=data['model'].split('/')[-1]
        for seed in seeds:
            for name,options in variants(profile):
                output=Path(out_dir)/model/task/f'seed{seed}_{name}.json'
                argv=['-m','experiments.run_rare_facts','--model',data['model'],'--revision',data['revision'],
                      '--manifest',str(path),'--out',str(output),'--device','cuda','--seed',str(seed),
                      '--selection','random,scan' if data.get('oracle_available') is False else 'oracle,random,scan',
                      '--calibration-dir',str(Path(out_dir)/'calibration'),'--resume','--stop-eos']
                for key,value in options.items():
                    argv.extend(['--'+key.replace('_','-'),str(value)])
                jobs.append(dict(name=f'{model}/{task}/seed{seed}/{name}',argv=argv,output=str(output)))
    outputs=[j['output'] for j in jobs]
    if len(outputs)!=len(set(outputs)):
        raise ValueError('plan output collision: use one manifest per model/task or separate plans')
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',nargs='+',help='Frozen manifests; shell wildcards are supported by your shell')
    p.add_argument('--profile',choices=('comparison','ablation'),default='comparison')
    p.add_argument('--out-dir',default='outputs/suite')
    p.add_argument('--seeds',default='3')
    p.add_argument('--plan',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    path=Path(args.plan)
    if args.manifests:
        jobs=make_jobs(args.manifests,args.profile,args.out_dir,tuple(map(int,args.seeds.split(','))))
        plan=dict(schema=1,profile=args.profile,jobs=jobs,
                  note='Seeds change codec/random selection, not frozen documents. Ablations are one-factor, not a Cartesian sweep.')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            raise FileExistsError('plan exists; choose a new path, or execute the existing plan without --manifests')
        path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    else:
        plan=json.loads(path.read_text(encoding='utf-8'))
    print(f"{len(plan['jobs'])} jobs in {path}. Execution requested: {args.execute}",flush=True)
    if args.execute:
        for job in plan['jobs']:
            print(job['name'],flush=True)
            # Explicitly executing a plan is authorization for these Python commands.
            subprocess.run([sys.executable,*job['argv']],check=True)


if __name__=='__main__':
    main()
