"""Dataset protocol adapted from the supplied ICLR-paper experiment code.

Only selected, reviewed definitions are included. See reference_sources.json.
Historical token IDs were not provided; freeze newly generated IDs for comparisons.
"""
import os
import inspect
import math
from pathlib import Path
import torch

TEXT_ROOT = Path(__file__).resolve().parent.parent / 'datasets' / 'reference_texts'


NAMES = ["project Alpha", "vault 7", "locker 12", "sample B-19", "shipment 44",
         "account Delta", "module Zeta", "batch 3", "unit Sigma", "archive 9"]


def build_filler_chunks(tok, approx_tokens=300, n_chunks=200):
    ids = []
    import os, inspect, argparse as _a, json as _j, difflib as _d, statistics as _s
    for p in ("kvtc_paper.txt", "proposal.txt"):
        if (TEXT_ROOT / p).exists():
            ids.extend(tok(open(TEXT_ROOT / p, encoding="utf-8").read(), add_special_tokens=False).input_ids)
    for m in (_a, _j, _d, _s):
        try:
            ids.extend(tok(inspect.getsource(m), add_special_tokens=False).input_ids)
        except Exception:
            pass
    chunks = [tok.decode(ids[i:i + approx_tokens]) for i in range(0, len(ids) - approx_tokens, approx_tokens)]
    return chunks[:n_chunks]


def make_doc(rng, tok, chunks, n_facts=4, n_chunks=6, target_len=1800):
    names = rng.sample(NAMES, n_facts)
    codes = [f"{rng.randint(100000, 999999)}" for _ in range(n_facts)]
    body = rng.sample(chunks, n_chunks)
    # plant facts between chunks, spread across the document
    slots = sorted(rng.sample(range(1, n_chunks), n_facts))
    parts = []
    for i, ch in enumerate(body):
        parts.append(ch)
        if i + 1 in slots:
            j = slots.index(i + 1)
            parts.append(f"\nNote: the reference code for {names[j]} is {codes[j]}.\n")
    text = "".join(parts)
    ids = tok(text, add_special_tokens=False).input_ids[:target_len]
    # make sure every fact survived truncation; if not, rebuild
    dec = tok.decode(ids)
    if not all(c in dec for c in codes):
        return make_doc(rng, tok, chunks, n_facts, n_chunks, target_len)
    # position of each fact as a fraction of the document
    pos = [dec.find(c) / max(len(dec), 1) for c in codes]
    return torch.tensor(ids), names, codes, pos


def question_for(tok, name):
    q = f"\nQuestion: What is the reference code for {name}?\nAnswer: The reference code for {name} is"
    return torch.tensor(tok(q, add_special_tokens=False).input_ids)


def fact_span(ids, tok, code):
    """[start, end) token span of the planted code, or (-1,-1)."""
    for cd in (tok(code, add_special_tokens=False).input_ids,
               tok(" " + code, add_special_tokens=False).input_ids):
        idl = ids.tolist()
        for st in range(len(idl) - len(cd) + 1):
            if idl[st:st + len(cd)] == cd:
                return st, st + len(cd)
    return -1, -1


STDLIB = ["argparse", "json", "difflib", "statistics", "textwrap", "collections", "functools",
          "inspect", "pathlib", "typing", "dataclasses", "logging", "csv", "calendar", "decimal",
          "fractions", "heapq", "pprint", "shutil", "tarfile", "zipfile", "subprocess", "threading",
          "configparser", "string", "random", "locale", "ftplib", "smtplib", "imaplib", "tokenize",
          "ast", "dis", "pickle", "copy", "enum", "gettext", "optparse", "platform", "tempfile",
          "glob", "fnmatch", "bisect", "queue", "sched", "uuid", "base64", "mailbox", "mimetypes",
          "cmd", "shlex", "trace", "profile", "pdb", "timeit", "doctest", "pydoc", "email.message",
          "http.client", "urllib.parse", "unittest.case", "xml.dom.minidom", "asyncio.base_events"]


def build_long_chunks(tok, approx=300):
    texts = []
    for p in ("kvtc_paper.txt", "proposal.txt", "arkvale_paper.txt", "mikv_paper.txt"):
        if (TEXT_ROOT / p).exists():
            texts.append(open(TEXT_ROOT / p, encoding="utf-8").read())
    import importlib
    for name in STDLIB:
        try:
            texts.append(inspect.getsource(importlib.import_module(name)))
        except Exception:
            pass
    chunks = []
    for t in texts:
        ids = tok(t, add_special_tokens=False).input_ids
        chunks += [tok.decode(ids[i:i + approx]) for i in range(0, len(ids) - approx, approx)]
    return chunks


def make_long_doc(rng, tok, chunks, target_len, n_facts=4, approx=300):
    for _ in range(50):
        n_chunks = int(target_len / approx * 1.15) + 4
        body = rng.sample(chunks, min(n_chunks, len(chunks)))
        names = rng.sample(NAMES, n_facts)
        codes = [f"{rng.randint(100000, 999999)}" for _ in range(n_facts)]
        usable = max(n_facts + 1, int(target_len / approx * 0.92))
        slots = sorted(rng.sample(range(1, usable), n_facts))
        parts = []
        for i, ch in enumerate(body):
            parts.append(ch)
            if i + 1 in slots:
                j = slots.index(i + 1)
                parts.append(f"\nNote: the reference code for {names[j]} is {codes[j]}.\n")
        ids = tok("".join(parts), add_special_tokens=False).input_ids
        if len(ids) < target_len:
            continue
        ids = torch.tensor(ids[:target_len])
        spans = [fact_span(ids, tok, c) for c in codes]
        if all(s >= 0 for s, _ in spans):
            return ids, names, codes, [s / target_len for s, _ in spans]
    raise RuntimeError("could not build a document with all facts inside the window")


def paraphrase_question(tok, name):
    """Reworded question: no 'reference code' phrase, so the digest cannot rely on lexical overlap."""
    q = (f"\nQuestion: Which six-digit number was assigned to {name} in the text above?"
         f"\nAnswer: The six-digit number assigned to {name} is")
    return torch.tensor(tok(q, add_special_tokens=False).input_ids)


def sentence_span(st, en, S, before=24, after=4):
    """Token span of the whole fact sentence (name + code), for oracle controls."""
    return max(0, st - before), min(S, en + after)


def load_corpus(tok, n_docs, ctx):
    texts = []
    for p in ("kvtc_paper.txt", "proposal.txt"):
        if (TEXT_ROOT / p).exists():
            texts.append(open(TEXT_ROOT / p, encoding="utf-8").read())
    import inspect, argparse as _a, json as _j, difflib as _d, statistics as _s
    for m in (_a, _j, _d, _s):
        try:
            texts.append(inspect.getsource(m))
        except Exception:
            pass
    ids = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False).input_ids)
    if len(ids) < n_docs * ctx:
        ids = ids * math.ceil(n_docs * ctx / len(ids))
    return [torch.tensor(ids[i * ctx:(i + 1) * ctx]) for i in range(n_docs)]

