# KVTC improvement experiments

The next implementation target is a **query-selected hot cache backed by a
recoverable, compressed cold cache**. KVTC remains the base codec. The earlier
wavelet proposal is not the current implementation target.

This project starts with a working copy of the supplied reproduction. We are
establishing numerical correctness before adding retrieval. The reported
40-document result is not reproduced by this repository, and there is no verified
novelty or GPU speed claim here.

## Start here

**Kaggle T4 small models:** [KAGGLE_T4.md](KAGGLE_T4.md) and
[ready-to-upload notebook](notebooks/kaggle_t4.ipynb) run shortened FP16/SDPA
experiments with Qwen2.5-0.5B/1.5B/3B, all eight methods, and per-example resume.

**Clone-to-results instructions:** [GETTING_STARTED_H200.md](GETTING_STARTED_H200.md)
walks through cloning, installation, authentication, all tests, native GPU smoke
tests, full execution, resume, judging and results. It also explains the runtime
budget and how to refine it from actual H200 measurements.

**Current four-model run:** [PHASE_ONE_RUNBOOK.md](PHASE_ONE_RUNBOOK.md) contains
the mentor's Linux terminal commands for LongBench **v2** and FreeKV's LongGenBench,
including Qwen2.5-72B, `--gpus 5` scheduling, per-example resume, separate Qwen3-32B
judging, and combined result tables. Run the GPU smoke matrix before the full suite.
Implementation differences and the current ours long-generation limitation are
recorded explicitly. This supersedes the earlier benchmark/model selection below.

**2026-09-18 H200 comparison update:** [H200_COMPARISON_RUNBOOK.md](H200_COMPARISON_RUNBOOK.md)
adds GPU-count-aware plans for Llama-3.1-8B and Qwen2.5-7B/14B, official KIVI/ArkVale
bridges in isolated environments, atomic resume, combined tables, LongBench-16
task scoring, and page-size sweeps. Compatibility gaps remain explicit: original
ArkVale does not support the requested models unchanged; KIVI has no Qwen2 port.
H200 execution and model sharding still need hardware validation. The registry
records the other requested baselines as pending, not implemented results.

**2026-09-17 evaluation update:** read [EVALUATION_RUNBOOK.md](EVALUATION_RUNBOOK.md)
for the local -> GitHub -> H200 workflow. The runner now includes grouped low-bit
quality controls, approximate eviction/recency baselines, reconstructed-cache
refinement, identity-checked calibration reuse, document-level resume and detailed
metrics. The dataset preparer supports complete splits of six English LongBench v1
QA tasks, and the suite generator creates comparison/one-factor ablation plans.
The local suite passed **82 tests**; the tiny random model completed **14 arms**
and a resume check. Qwen2.5 7B/14B GPU runs remain unexecuted. These are quality
controls and timing telemetry, not official baseline implementations or an isolated
equal-memory serving benchmark.

Checks completed locally on 2026-09-16: **73 tests passed** with model dependencies
installed. The indexed archive now has a Qwen2 sparse-cache adapter, the reference
rare-fact dataset generator, and a stored-key-coefficient scan selector baseline.
The seven-arm runner completed on a tiny random-weight model. Pretrained task
accuracy and H200 performance have NOT been measured in this project.

Read `reports/reference_implementation_comparison.md` for the comparison with the
new ICLR-paper folder, and `H200_RUNBOOK.md` for the prepared model experiments.
The comparison includes important limits in the reference's full-decode/masking
path and its unquantized-key selector. Our paging implementation remains the base.

The local `.venv` is isolated from the source folder and uses CPU PyTorch for
small checks. From this project directory, run in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.smoke
.\.venv\Scripts\python.exe -m scripts.selective_decode
.\.venv\Scripts\python.exe -m scripts.storage_overhead
```

The second command performs calibration, serialization and reconstruction on
**synthetic** data and writes `outputs/smoke.json`. It does not download a model.
Its ratios and errors must not be reported as research results.

To create the environment elsewhere, use Python 3.12 and install the versions in
`requirements-cpu.txt` into a virtual environment. A future GPU environment needs
a CUDA-capable PyTorch build appropriate for its driver; do not reuse the CPU
build for GPU measurements.

## The implementation stages

| Stage | What to build | Evidence required to proceed |
| --- | --- | --- |
| 1 Baseline correctness | Fix numerical edge cases; test PCA, RoPE, DP, packing and decoding | All tests pass; stored-byte decode matches the simulated quantizer |
| 2 Addressable cold storage | Encode independent token pages with a shared calibration artifact and an explicit byte index | Decode a chosen page without decoding other pages; match full decoding of the same page-based archive |
| 3 Oracle retrieval | Supply the known relevant page from the planted-fact dataset, plus a fixed hot set | The model can recover the answer when retrieval is perfect; otherwise fix codec/inference before training a selector |
| 4 Actual selection | Rank pages using incoming layer/head queries and accessible compressed representations | High answer-relevant page recall at the target page budget on held-out documents |
| 5 Fair comparisons | Eviction, full decode of the same cold archive, original monolithic KVTC, low-precision retention and random-page controls | Paired quality results with equal total byte budgets and uncertainty |
| 6 GPU execution | Move paging, ranking and reconstruction to a measured GPU path | End-to-end timing includes selection, entropy decoding, transfers, reconstruction and attention |

Stages 1 and the CPU storage portion of stage 2 are implemented. The model adapter
and stages 3/4's experiment runner are now implemented and checked on a tiny random
model. Next run pretrained Qwen quality controls on the H200. The current selector
reads the independently stored key-head stream and allocates a coefficient buffer;
it is not yet the efficient GPU-only final system. Local grouped low-bit controls and timing telemetry
are now present; official/validated packed baseline comparisons and isolated timing
remain pending.

### The first cold-store interface

Implemented in `kvtc/cold_store.py`:

```python
from kvtc.cold_store import ColdStore

archive = ColdStore.encode(codec, keys, values, page_tokens=128)
selected = archive.decode_pages([3, 7])
full = archive.decode_all()
```

The decoder returns `selected.positions`, `selected.keys`, `selected.values`,
`selected.payload_bytes_read`, and the archive's separately counted `index_bytes`.
It deduplicates page IDs and returns tokens in original document order. Reopen
stored bytes with `ColdStore(archive.blob, codec)`, supplying the same calibration
artifact. Artifact identity is the caller's responsibility; only feature dimension
is checked. The archive holds no original K/V tensors. Shared calibration tensors
must not be modified in place while decoding an archive.

PCA, quantizers, DP assignments and global protection regions are reused. The
storage format changes: every page has independent K/V entropy streams and an
index entry. No changes were made to the existing codec for paging. GPU entropy
decoding is not implemented; this prototype retains the codec's CPU serialization
and CPU output tensors.

The default compact v3 archive stores shared metadata once, uses 64-bit page
offsets with implicit token ranges, and splits each page's key codes into an
independently readable leading coefficient stream (256 by default) and a tail.
Scoring reads the head, key quantization metadata and protected keys; selected-page
reconstruction reads both key streams and the value streams. No coefficient is
duplicated. The v1 and v2 formats remain readable and can be encoded explicitly.
Reconstructed values are preserved. At decoding time,
the prototype rebuilds transient legacy headers for selected pages to reuse the
existing numerical decoder; that allocation work has not been timed.

On the original synthetic example with unchanged 128-token pages, the archive
shrank from 22,553 bytes to 17,748 bytes. Overhead versus the original 16,458-byte
monolithic codec fell from 37.03% to 7.84%. Versus a monolithic one-page archive
using the same compact metadata, overhead is 10.39%. Shared calibration is excluded
from all sizes. The selective demo reads 4,113 of 17,493 page-payload bytes to
recover 256 of 1,024 tokens; the shared metadata/index costs 255 bytes.

See `reports/storage_overhead.md` for a 15-setting page-size/data sweep and byte
breakdown. Larger pages reduce storage overhead but recover more unrelated tokens.
These are synthetic correctness/storage measurements, not model accuracy or latency.

`decode_pages` does not automatically include protected tokens or a hot set.
The new experiment runner includes protected tokens in its hot set; its adapter
deduplicates overlaps and uses original positions for RoPE and causal masking. The attention
formula can stay the same, but excluding entries changes its softmax normalization
and potentially its output. Matching a selected page's full-decode values does
not imply matching full-cache attention or answer accuracy.

`selected` must return the **original token positions**, reconstructed K/V, and
actual counts of bytes read and pages decoded. Keep global sink/window protection
consistent; do not accidentally protect another 132 tokens inside every page.

Store each page in an independently decodable entropy stream. Count page headers,
offsets, scales, shifts, protected entries and the hot cache. Report shared basis
cost separately and explicitly when amortizing it. A byte stream without an index
does not provide random access just because its PCA rows are independent.

Compare selected-page decoding with full decoding of the **same archive** for
correctness. Separately compare page-based storage with monolithic KVTC to expose
the compression overhead introduced by paging.

### Before using a language model

The inference adapter must preserve original positions, causal masks, chat
formatting and KV head grouping. Merge hot and recovered entries by token position
without duplicates. A recovered entry is still lossy even if stored in FP16.
Decide when the hot set changes and whether selection runs per query token, layer
or head; a question is not represented by one universal attention query.

For the initial oracle experiment, compress the document **before** showing the
question, and restore a fresh cache for every method/question. Keep calibration,
validation and final test documents separate. Record individual answers, not just
an aggregate percentage. No accuracy targets are assumed already achieved.

### Selector representations

Quantized PCA coefficients, dequantized PCA coefficients, and DEFLATE bytes are
different representations. Specify which the selector reads. Ordinary arithmetic
cannot score an entropy-coded stream directly. Any stored summaries or decoded
coefficient buffers must be counted in memory, and any scan/decode work must be
included in latency. Start with a simple selector; optimize only after the oracle
experiment shows recoverability.

## What to read in the code

Read `scripts/smoke.py` first: it calls the entire baseline pipeline in order.
Then read `codec.py` in `kvtc/`, followed by `pca.py`, `quant.py`, `dp.py`, and
`serialize.py`. `tests/test_baseline.py` provides small numerical examples.

The supplied source is recorded in `PROVENANCE.md`. Its original README is kept
under `work/README_supplied.md` for local reference; statements in that README are
not independent verification of the reproduction.
