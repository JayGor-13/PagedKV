"""Inspect a proposed larger Qwen run without downloading model weights."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--revision',required=True)
    p.add_argument('--rank-cap',type=int,default=8192)
    p.add_argument('--calib-tokens',type=int,default=24576)
    p.add_argument('--context',type=int,default=32768)
    p.add_argument('--out',default='outputs/preflight.json')
    args=p.parse_args()
    if min(args.rank_cap,args.calib_tokens,args.context)<1:
        p.error('positive sizes required')
    from transformers import AutoConfig
    cfg=AutoConfig.from_pretrained(args.model,revision=args.revision)
    if cfg.model_type not in ('qwen2','llama') or getattr(cfg,'use_sliding_window',False):
        raise ValueError('current adapter supports Qwen2/Llama full-attention models only')
    if args.context>cfg.max_position_embeddings:
        raise ValueError('context exceeds configured capacity; do not silently enable unsupported RoPE scaling')
    d=getattr(cfg,'head_dim',cfg.hidden_size//cfg.num_attention_heads)
    dim=cfg.num_hidden_layers*cfg.num_key_value_heads*d
    rank=min(dim,args.rank_cap)
    result=dict(model=args.model,revision=getattr(cfg,'_commit_hash',args.revision),
        architecture=cfg.model_type,layers=cfg.num_hidden_layers,kv_heads=cfg.num_key_value_heads,
        feature_dim=dim,rank=rank,cuda_available=torch.cuda.is_available(),
        components_bytes=dict(one_fp64_gram=8*dim*dim,one_fp64_eigenvector_matrix=8*dim*dim,
            two_fp32_bases=2*4*dim*rank,calibration_K_and_V_cpu=2*4*args.calib_tokens*dim,
            one_projected_calibration_matrix=4*args.calib_tokens*rank,
            document_native_K_and_V_bf16=4*args.context*dim),
        warning='Components are not an upper bound: eigensolver workspace, model weights, replicated diagnostic caches, activations and allocator overhead are additional. No fit guarantee.')
    if torch.cuda.is_available():
        result['gpu']=torch.cuda.get_device_name(0)
        result['free_cuda_bytes'],result['total_cuda_bytes']=torch.cuda.mem_get_info()
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
