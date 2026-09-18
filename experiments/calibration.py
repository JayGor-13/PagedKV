"""Reusable tensor-only calibration artifacts, keyed by data/model/config identity."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from kvtc import Basis, CalibrationArtifact, Assignment


def identity(model, revision, config, calibration, tokenizer_sha256=None):
    value = dict(model=model, revision=revision, config=asdict(config),
                 calibration=calibration, tokenizer_sha256=tokenizer_sha256)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load(codec, path, signature):
    data = torch.load(path, map_location=codec.device, weights_only=True)
    if data['signature'] != signature:
        raise ValueError('calibration artifact identity mismatch')
    bases = {name: Basis(**data[name]) for name in ('key', 'value')}
    assignments = {name: Assignment(**data['assignments'][name]) for name in bases}
    codec.art = CalibrationArtifact(bases['key'], bases['value'], assignments, codec.cfg, data['p'])


def save(codec, path, signature):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(signature=signature, p=codec.art.p,
                assignments={k: asdict(v) for k,v in codec.art.assignments.items()})
    for name in ('key', 'value'):
        b = getattr(codec.art, name)
        data[name] = {k: getattr(b,k).detach().cpu() for k in ('mu','V','evals')}
    temporary = path.with_suffix('.tmp')
    torch.save(data, temporary)
    temporary.replace(path)
