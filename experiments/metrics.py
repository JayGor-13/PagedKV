"""Quality metrics and explicitly scoped runtime telemetry (not serving benchmarks)."""
from collections import Counter
import math
import re
import string
import time

import torch


def normalize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_scores(prediction, answers):
    """English QA token F1: same normalization/overlap convention as LongBench v1.

    Independently implemented; max over reference answers. Empty overlap gives F1=0.
    https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py
    """
    if not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
        raise ValueError('nonempty reference answers required')
    pred = normalize(prediction)
    tokens = pred.split()
    f1, em = 0., 0.
    for answer in answers:
        ref = normalize(answer)
        other = ref.split()
        overlap = sum((Counter(tokens) & Counter(other)).values())
        if overlap:
            f1 = max(f1, 2. * overlap / (len(tokens) + len(other)))
        em = max(em, float(pred == ref))
    return dict(exact_match=em, token_f1=f1,
                contains_target=any(a in prediction for a in answers))


def wilson(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    rate = correct / total
    center = (rate + z*z/(2*total)) / (1+z*z/total)
    half = z*math.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/(1+z*z/total)
    return [max(0., center-half), min(1., center+half)]


def sync(device):
    if str(device).startswith('cuda'):
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def measured(fn, device):
    sync(device)
    start = time.perf_counter()
    result = fn()
    sync(device)
    return result, (time.perf_counter()-start)*1000


def first_token_divergence(reference, candidate):
    p = reference.float().log_softmax(-1)
    q = candidate.float().log_softmax(-1)
    return dict(first_token_kl=float((p.exp()*(p-q)).sum().clamp_min(0)),
                first_token_top1_agrees=bool(p.argmax() == q.argmax()))
