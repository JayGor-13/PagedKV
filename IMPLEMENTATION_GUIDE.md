# Paged KVTC with query-selected cache recovery

Implementation guide — 17 September 2026

Review update — 17 September 2026: Sections 1–16 describe the **primary paged implementation**. Section 17 inventories additional implementations and saved experiments in the separate **ICLR paper** reference folder. A feature present there is not automatically integrated here. See [the attachment review](reports/attached_additions_review.md) for verified qualifications and missing evidence.

Implementation update later on 17 September: the primary runner now includes additional baseline controls, reconstructed-cache refinement and evaluation infrastructure. Section 18 and [EVALUATION_RUNBOOK.md](EVALUATION_RUNBOOK.md) describe the new implementation and supersede the earlier integration-status snapshot in Section 17.

This document describes the code currently implemented in **KVTC improvement**. It explains the system from basic concepts through its equations, storage format, model integration, tests and next experiments. It documents this reproduction; it is not a claim that every detail or result of the KVTC paper has been independently reproduced.

## 1. What are we building?

The objective is to retain a compressed backup of a document's entire KV cache, while letting the model use a smaller active cache when answering a question.

The system has two cache tiers:

- **Hot cache:** a selected set of original model-precision keys and values, immediately usable by attention.
- **Cold archive:** a KVTC-compressed copy of every document token's keys and values, divided into independently decodable pages.

When a question arrives, a selector ranks the cold pages. We reconstruct the selected pages, merge their entries with the hot cache, and let the model answer using that smaller active cache.

**Current limitation:** the selector scans all stored key streams and materializes a coefficient buffer. Only selected pages receive full K/V reconstruction, but it is incorrect to say that unselected pages are never read or partially decoded. Their values remain compressed during selection.

The project is a correctness and quality prototype. Pretrained-model accuracy, H200 latency and a fully GPU-resident execution path have not been established locally.

## 2. Vocabulary

| Term | Meaning in this implementation |
| --- | --- |
| Token | A tokenizer-produced piece of text, not necessarily a whole word. |
| Query, Q | A vector produced by an attention head describing what the current token is looking for. |
| Key, K | A stored vector used to match a query against a previous token. |
| Value, V | The information mixed into the attention output after query–key matching. |
| KV cache | Keys and values retained from previously processed tokens. |
| Prefill | Processing the document to build its cache. |
| PCA basis | Shared directions used to represent cache features in another coordinate system. |
| PCA coefficient | A token's coordinate along one PCA direction. |
| Quantization | Approximating numerical coefficients using fewer bits. This introduces error. |
| Entropy coding | Losslessly compressing the quantized representation, currently using CPU DEFLATE by default. |
| Page | A fixed contiguous range of original token positions. |
| Quantization block | A contiguous range of PCA-coordinate columns assigned one quantizer. |
| Hot token | An original cache entry retained in the active cache. |
| Halo | Neighboring pages included around a selected page. |
| Oracle selector | A diagnostic selector given the known answer location. Not a deployable method. |

## 3. End-to-end flow

```text
Calibration documents
    -> collect cache features
    -> fit shared K and V PCA bases
    -> choose quantization blocks and precisions using DP
    -> reusable calibration artifact

Evaluation document, before seeing its questions
    -> model prefill
    -> choose hot token positions
    -> remove RoPE from keys and flatten cache features
    -> divide token rows into pages
    -> shared PCA projection, quantization and independent entropy streams
    -> indexed cold archive
    -> scan stored key coefficients once for the scan selector

Question
    -> preliminary model pass using hot cache
    -> capture attention query vectors
    -> estimate token relevance from stored key coefficients
    -> aggregate relevance into page scores
    -> select pages and optional neighbors
    -> reconstruct selected full K/V
    -> merge with hot entries using original positions
    -> process question again and generate answer
```

## 4. Cache layout and RoPE

For each model layer, native keys and values have shape:

```text
[batch, number_of_KV_heads, number_of_tokens, head_dimension]
```

The present adapter supports batch size one and Qwen2 full-attention models. It concatenates features across all layers and KV heads to produce:

\[
K,V\in\mathbb{R}^{N\times p},\qquad p=L H_{KV}d.
\]

Each row still represents one original token. For the reference Qwen configuration, 28 layers, 2 KV heads and head dimension 128 give `p = 7168`.

### What RoPE does

Rotary positional embeddings rotate pairs of query/key coordinates by position-dependent angles. For one coordinate pair:

\[
\begin{bmatrix}x'_1\\x'_2\end{bmatrix}
=
\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}.
\]

Different coordinate pairs use different frequencies. Applying these rotations to Q and K makes attention sensitive to relative position. Values are not rotated by this operation.

The adapter removes RoPE from native keys before PCA compression. After selected keys are reconstructed, it reapplies RoPE at their **original document positions**. Recovered token 640 must retain position 640 even if it becomes the 100th entry in the smaller active cache.

Source: [model_adapter.py](experiments/model_adapter.py), [capture.py](kvtc/capture.py), [rope.py](kvtc/rope.py).

## 5. Calibration: what is learned before compression?

Calibration fits a separate mean and PCA basis for K and V. A key basis is shared by every key page; a value basis is shared by every value page. We do not learn a basis per page or per question.

For one cache type, let calibration rows be `X`. Center the rows and find dominant directions of variation:

\[
\mu=\operatorname{mean}(X),\qquad Z=(X-\mu)U.
\]

Here `U` has shape `[p, r]`, with directions ordered by decreasing variance. Reconstruction is:

\[
\widehat X=\widehat ZU^T+\mu.
\]

With all directions and no quantization, an orthonormal full basis permits reconstruction up to numerical error. Truncation and quantization introduce loss.

PCA changes the representation of each token; it does not move tokens between rows or identify relevant pages. High calibration variance also does not guarantee high relevance to a future question.

The code supports Gram-matrix eigendecomposition and randomized PCA. The Gram path requires memory proportional to `p²`; calibration also retains chunks and concatenates projected data. It is not a constant-memory calibration implementation.

### Dynamic programming chooses where to spend bits

After projection, the DP divides the PCA coordinates into contiguous blocks and assigns each block one of:

| Quantizer | Coordinate payload | Interpretation |
| --- | --- | --- |
| `none` | 0 bits | Reconstruct these coefficients as zero. |
| `int2` | 2 bits | Coarse quantization. |
| `int4` | 4 bits | Finer quantization. |
| `fp8` | 8 bits | An 8-bit floating-point quantization option. |

For a coded block of width `w` and precision `b`, the default per-token cost is:

\[
\operatorname{cost}=wb+32.
\]

The extra 32 bits store a 16-bit scale and 16-bit shift per block per token. A `none` block has zero payload and zero scale/shift cost.

For target compression ratio `CR`, the per-cache-type budget is approximately:

\[
B=\operatorname{round}(16p/CR)\quad\text{bits per token}.
\]

Conceptually, the DP solves:

\[
\min_{\text{blocks and quantizers}}\sum_b
\|Z_b-\widehat Z_b\|_F^2
\quad\text{subject to}\quad\sum_b\operatorname{cost}(b)\le B.
\]

It estimates errors from calibration coefficients, optionally subsampled. It optimizes reconstruction error, not question-answering accuracy or attention-output error. The chosen block assignments are reused during compression.

The core config defaults to `dp_stride=1`, searching every boundary. The model runner defaults to `--dp-stride 16` to restrict calibration work. That restriction can change the solution; rank must be divisible by the stride.

The target ratio is a bit-allocation target before entropy coding. It is not a promise of that ratio for the complete serialized archive, protected tokens, hot cache and shared basis.

Source: [codec.py](kvtc/codec.py), [pca.py](kvtc/pca.py), [dp.py](kvtc/dp.py), [quant.py](kvtc/quant.py), [config.py](kvtc/config.py).

## 6. Compression and page creation

For page size `P`, token `i` belongs to:

\[
\operatorname{pageID}(i)=\lfloor i/P\rfloor.
\]

Page `j` contains the half-open token range `[jP, min((j+1)P, N))`.

For 1,024 tokens and 128-token pages:

| Page | Token positions, inclusive |
| --- | --- |
| 0 | 0–127 |
| 1 | 128–255 |
| 2 | 256–383 |
| 3 | 384–511 |
| 4 | 512–639 |
| 5 | 640–767 |
| 6 | 768–895 |
| 7 | 896–1023 |

Pages have numerical IDs, not semantic labels. They may split sentences. There is no clustering of similar PCA vectors.

Conceptually, encoding does:

```python
for start in range(0, N, page_size):
    end = min(start + page_size, N)
    # Reuse the calibration artifact and compute protection globally.
    encode_page(K[start:end], V[start:end])
```

For unprotected rows, each page performs PCA projection, quantization, bit packing and lossless entropy compression. The protected rows bypass lossy PCA quantization and are stored in 16-bit form; those bytes can still be losslessly entropy-compressed.

### Protection happens once per document

The default protected set is the first 4 and last 128 document tokens. Each page receives only its overlap with those global ranges. A page does not receive a new 4-token sink and 128-token window just because it is independently encoded.

Short documents whose protected ranges cover all rows are entirely protected. Small pages in the middle of a long document can still be compressed.

Source: [cold_store.py](kvtc/cold_store.py), particularly `ColdStore.encode` and its protection logic.

## 7. Archive format and random access

The default compact v2 archive contains:

1. A fixed prefix identifying the format and sizes.
2. One compressed shared metadata record: sequence length, feature dimension, page size, global protection and quantization layout.
3. An array of 64-bit page offsets, including an end offset.
4. Independently compressed page payloads.

Each page has a 25-byte binary prefix with flags and six stream lengths: key codes, key scale/shift metadata, key protected rows, and the corresponding three value streams. Empty streams are omitted. Token ranges are inferred from page IDs and page size.

PCA tensors are supplied through the shared calibration artifact, not repeated in page payloads. The caller must supply the matching artifact when reopening an archive. Feature dimension is checked, but artifact identity is not cryptographically validated.

```python
from kvtc.cold_store import ColdStore

# codec must already be calibrated; keys/values have shape [N, p].
archive = ColdStore.encode(codec, keys, values, page_tokens=128)
selected = archive.decode_pages([2, 5])
full = archive.decode_all()
reopened = ColdStore(archive.blob, codec)
```

`decode_pages` validates, deduplicates and sorts IDs, then jumps to the indexed payloads. It does not decode pages 0–4 to reach page 5. It returns reconstructed K/V, original positions, page IDs and selected payload bytes read; shared index bytes are accounted separately.

The decoder reconstructs temporary legacy headers in memory to reuse the base codec. These headers are not stored repeatedly in v2. The earlier v1 archive remains supported.

The required numerical control is:

> Decoding a selected page alone must match the same token rows obtained by fully decoding that same archive.

This does not imply that a model using a subset of tokens matches full-cache attention. Omitting tokens changes the attention softmax and potentially the answer.

## 8. How the hot cache is created

The runner collects document-side attention queries during prefill and computes an approximate H2O-style importance score. The hot policy combines sink/recent tokens and highly scored positions. It then explicitly includes the document's protected first 4 and last 128 tokens.

The resulting hot set can be larger than the requested `--hot-tokens` value, so actual counts and bytes must be used. Native model K/V entries are physically sliced to create the hot cache. This is not an independently validated implementation of the official H2O algorithm.

The cold archive retains every document token, including hot positions. This duplication must be counted. When recovery overlaps a hot position, the original hot entry wins.

## 9. How automatic page selection works

### 9.1 Prepare a coefficient buffer from stored keys

`scan_key_coefficients` visits every page and entropy-decodes its key streams. It dequantizes the first `c` PCA coordinates needed by the selector; coordinates assigned `none` stay zero. Protected keys have no stored PCA codes, so their stored 16-bit values are projected into the shared basis.

The result is a CPU FP32 buffer:

\[
\widetilde Z_K\in\mathbb{R}^{N\times c},\qquad M=4Nc\text{ bytes}.
\]

For `N=32768` and `c=256`, that buffer is 32 MiB. It is additional working memory even though it is not added to the archive on disk. The runner prepares it once per document and reuses it for that document's questions.

The scan reads stored quantized keys, not original unquantized cache features, and does not read the value streams. Ordinary arithmetic is not performed directly on DEFLATE bytes.

### 9.2 Obtain query vectors from the model

`question_forward(..., collect_queries=True)` runs the question using the hot cache and captures pre-RoPE query projection outputs at the attention layers. Each layer and head has its own queries; there is no single universal question vector.

This preliminary pass does not know the selected pages. The model supplies query vectors; our external selection code decides the page IDs. No separate routing model is trained.

### 9.3 Estimate relevance and aggregate by page

For layer `l` and KV head `h`, recover an approximate key matrix from the limited coefficients:

\[
\widehat K_{l,h}=\widetilde Z_K U_{K,l,h,:c}^{T}+\mu_{K,l,h}.
\]

Only the corresponding feature slice of the shared basis is used. For each question query `q`:

\[
s_i=q^T\widehat k_i/\sqrt d,\qquad
a_i=\exp(s_i)/\sum_{j=0}^{N-1}\exp(s_j).
\]

Page relevance is the sum of these weights for non-hot entries in the page, accumulated over the chosen layers, query heads and question tokens:

\[
R(P)=\sum_{l,h,t}\sum_{i\in P,\ i\notin\mathrm{hot}}a_{l,h,t,i}.
\]

Normalization includes all document tokens; hot entries are excluded only from the page contribution. Grouped-query attention heads are matched to their corresponding shared KV head.

The approximation uses **pre-RoPE** Q and K. It ignores positional rotation when ranking. Actual generation uses correctly positioned keys after recovery.

The selector temporarily reconstructs approximate keys across all tokens for each scored layer/head. Therefore, this is a linear key scan, not a relevance tree, approximate nearest-neighbor index or direct lookup of an already-known page.

### 9.4 Select pages and optional neighbors

The code sorts page scores and takes the first `recall_k` page IDs. A halo includes neighboring pages, and the union is deduplicated.

For example, selecting page 5 with `halo=1` recovers pages 4, 5 and 6. Selecting eight pages can therefore decode more than eight pages.

| Runner option | Meaning | Default |
| --- | --- | --- |
| `--page` | Tokens per page | 128 |
| `--topk` | PCA coordinates used for ranking, not number of pages | 256 |
| `--recall-k` | Highest-ranked pages before halo expansion | 8 |
| `--halo` | Neighbor radius | 1 |
| `--band` | Zero-based layer interval, upper endpoint excluded | `14,21`, layers 14–20 |

Source: [selector.py](experiments/selector.py), [reference_heuristics.py](experiments/reference_heuristics.py), [run_rare_facts.py](experiments/run_rare_facts.py).

## 10. Recovering selected pages and generating an answer

`recover` calls `decode_pages` only on the selected IDs. For these pages it reverses entropy coding, bit packing, quantization and PCA projection:

\[
\widehat K=\widehat Z_KU_K^T+\mu_K,
\qquad
\widehat V=\widehat Z_VU_V^T+\mu_V.
\]

Full reconstruction means full feature dimensions, not lossless recovery of the original tensors.

Recovered entries already present in the hot cache are removed. The adapter restores native layer/head layout, applies RoPE at original positions, combines cold and hot entries, and sorts them by original token position.

\[
\text{active positions}=\text{hot positions}\cup\text{recovered positions}.
\]

The question is processed again using a fresh cache built from these active entries. If the document had `N` tokens, question positions begin at `N`, irrespective of the shorter physical cache length. Generation proceeds with the original positional timeline and appends new entries normally.

By default there is one selection pass. `--refine N` now enables N additional question-query passes over the reconstructed active cache, followed by reranking and real page recovery. Each round is reported separately with cumulative recovery cost. The final retrieved set is fixed during answer generation. Per-generated-token reranking and a continuously maintained hot/cold eviction policy are not implemented.

## 11. Experiment runner and dataset

The adapted task generator creates documents with planted rare facts using bundled paper/proposal text and Python standard-library text. It is a synthetic retrieval task, not LongBench or RULER.

The runner freezes tokenized documents, calibration inputs and questions into a manifest. Reusing the manifest gives paired inputs across methods. An existing manifest overrides new dataset-generation flags such as seed, context length and document count; use a new filename to intentionally change the dataset.

The document is compressed before its question is processed. Each evaluation gets a fresh model cache so question/generation mutations do not contaminate another arm.

| Arm | Purpose |
| --- | --- |
| `vanilla` | Original complete cache. |
| `monolithic_kvtc_full` | Full reconstruction using the monolithic codec. |
| `paged_kvtc_full` | Full reconstruction of the page-based archive. |
| `hot_only` | What survives without cold retrieval. |
| `oracle` | Recovery using known fact/sentence locations; diagnostic only. |
| `random` | Random page selection control. |
| `scan` | Actual stored-key-coefficient page selector. |

The expanded runner also supports `h2o_approx`, `snap_approx`, `recent_approx`,
`kivi4_local_full`, `kivi2_local_full`, `int4_local_full`, and optional `scan_refineN`.
See Section 18 for their scope. The original seven-arm configuration is available
with `--baselines none --refine 0`.

The reference success rule checks whether the target string appears in the generated answer. This is not a strict whole-answer exact-match metric. Exact-style and reworded questions are inherited from the reference protocol; do not assume they form a controlled same-fact paraphrase pair.

Frozen historical reference token IDs were unavailable. The adapted protocol is reproducible with its own manifests, but cannot promise identical historical documents. Calibration and evaluation filler sources also overlap; stricter held-out-source experiments remain necessary.

Full diagnostic caches coexist in this quality harness. Its total process peak memory and runtime are not isolated per-method deployment measurements.

## 12. Tests and evidence already available

The recorded local check on 16 September 2026 completed with **73 tests passed**, including model dependencies. A tiny random-weight Qwen2 smoke run completed all seven experiment arms. These are previous recorded checks, not new tests run to write this document.

After the 17 September evaluation update: **82 tests passed**, and a new tiny-model
smoke run completed all 14 configured arms plus calibration-reuse/resume validation.
This remains correctness evidence, not pretrained task quality.

| Test file | Controls covered |
| --- | --- |
| [test_baseline.py](tests/test_baseline.py) | Quantizer edge cases; DP error/budget controls; full-rank PCA; RoPE round trip; mixed-width packing; serialized reconstruction. |
| [test_cold_store.py](tests/test_cold_store.py) | Global protection; indexed selected-page decoding; invalid IDs; compact/legacy agreement; unselected corrupt-page isolation. |
| [test_document_page_controls.py](tests/test_document_page_controls.py) | Shared calibration, global first-4/last-128 protection and exact isolated-page/full-archive agreement. |
| [test_model_adapter.py](tests/test_model_adapter.py) | All-page model-logit control; sparse-vs-masked position control; selective recovery and hot precedence; key-only scan matching stored quantization. |
| [test_reference_task.py](tests/test_reference_task.py) | Deterministic generation, planted facts and frozen-manifest reuse. |

Tiny random-model tests validate tensor layout, cache wiring and numerical controls. Their generated answers cannot establish language-model task accuracy.

### Recorded synthetic storage result

For the original 1,024-token synthetic example with 128-token pages:

| Representation | Bytes |
| --- | ---: |
| Original monolithic codec | 16,458 |
| Earlier v1 page archive | 22,553 |
| Compact v2 page archive | 17,748 |

Overhead relative to the original monolithic codec fell from 37.03% to 7.84%. Relative to a one-page archive using the same compact metadata, overhead is 10.39%. Shared calibration is excluded from these sizes.

The selective demo recovers 256 of 1,024 tokens by reading 4,113 of 17,493 page-payload bytes, with 255 bytes of shared metadata/index. This describes selected-page decoding alone, not the automatic selector's additional key scan.

These are synthetic measurements, not an expected compression ratio or speedup on Qwen. See [storage_overhead.md](reports/storage_overhead.md) and [page_controls.md](reports/page_controls.md).

## 13. Running the checks

Run from a project root with a configured Python environment. The known local Windows virtual environment is in the OneDrive working copy, not necessarily beside the Desktop copy.

```powershell
Set-Location 'C:\Users\jaygo\OneDrive\Documents\ChatGPT\KVTC-Improvement'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.smoke
.\.venv\Scripts\python.exe -m scripts.selective_decode
.\.venv\Scripts\python.exe -m scripts.audit_page_controls
.\.venv\Scripts\python.exe -m scripts.storage_overhead
.\.venv\Scripts\python.exe -m scripts.smoke_model_integration
```

On the H200 machine, use its CUDA-capable PyTorch environment. Do not copy the Windows CPU virtual environment or install the CPU-only torch requirements into the GPU environment.

```bash
python -m pip install -r requirements-model.txt
python -m pytest -q
python -m scripts.smoke_model_integration

python -u -m experiments.run_rare_facts \
  --model Qwen/Qwen2.5-1.5B-Instruct --device cuda \
  --protocol short --ctx 1800 --docs 4 --seed 1 \
  --page 128 --hot-tokens 256 --recall-k 2 --halo 1 \
  --selection oracle,random,scan \
  --manifest outputs/short_seed1_4docs.json \
  --out outputs/short_seed1_p128_results.json
```

The first pretrained run may download the model/tokenizer. Default calibration is 12 documents of 2,048 tokens with a 7,168-feature model cache: this is not the tiny CPU smoke workload. Pin model revision and record package, driver and GPU details for archival experiments.

See [H200_RUNBOOK.md](H200_RUNBOOK.md) for the 16K/32K protocol and manifest commands.

## 14. What remains to be implemented or established?

| Item | Current status / required work |
| --- | --- |
| Independent indexed page storage | Implemented and locally tested. |
| Shared basis and document-wide protection | Implemented and locally tested. |
| Physical sparse model cache and original positions | Implemented for Qwen2 full attention, tested with a tiny random model. |
| Oracle/random/scan quality harness | Implemented; pretrained H200 runs pending. |
| Efficient selection without a full key-stream scan | Not implemented. Current baseline materializes coefficients and approximate per-head keys. |
| GPU entropy coding/decoding | Not implemented; current default serialization uses CPU zlib. |
| Low-precision retention baselines | Local grouped K4/V2, K2/V2 and K4/V4 full-retention quality controls are integrated. Official packed kernels and validated equal-memory retained subsets remain pending. |
| Exact equal-total-memory comparisons | Pending. The reference improves accounting by including recalled active entries, but does not complete total-memory accounting. |
| End-to-end GPU timing | Pending for the primary paged system. Reference component timings and synthetic decode-scaling studies exist; see Section 17. |
| Adaptive retrieval while generating | Not implemented. Optional pre-generation refinement is available. |
| Broader model families and batching | Not implemented by the current adapter. |
| Pretrained quality, reliable speedup and novelty claims | Not established by local tests. |

Memory comparisons must include the cold archive, index, hot cache, recovered active cache, selector buffers and temporary tensors. Report actual shared basis storage separately and state any amortization assumption. Include question prepass, scans, transfers, entropy decode, inverse PCA, merging and generation in end-to-end timing.

## 15. Experiments and conditions for moving forward

These are proposed decision rules, not completed results or statistically chosen thresholds.

1. **Validate full-cache controls on the pretrained model.** Check vanilla, monolithic KVTC and full paged KVTC. Investigate substantial unexplained quality differences before evaluating selective recall.
2. **Validate oracle recovery.** If known relevant pages plus the hot cache cannot recover answers that full KVTC answers, investigate context requirements, compression and inference integration before improving the selector.
3. **Compare scan against random retrieval.** Measure fact-span coverage and answer quality at the same actual recovered-token/page budgets, including halo expansion. The scan should provide repeatable improvements on held-out documents.
4. **Sweep retrieval parameters on validation data.** Vary page size, coordinate count, layer band, halo and retrieval budget. Freeze choices before final evaluation. Track cases where a relevant page is found but its answer still fails.
5. **Add fair memory baselines.** Compare against low-precision retention and eviction with complete byte accounting. A larger total memory budget cannot be credited as a method improvement.
6. **Measure isolated end-to-end performance.** Determine whether attention savings exceed query-prepass, key-scan, decode and transfer costs. Report first-question and repeated-question costs separately because the coefficient buffer is reusable.
7. **Broaden the workload.** Evaluate held-out sources and more realistic question types. Planted facts alone do not establish general long-context reasoning quality.

A positive research result would combine strong held-out quality with a measured benefit in total memory, latency or both. Fewer attended tokens and fewer fully decoded pages are intermediate measurements, not proof of a speedup.

## 16. Code reading map

| File | Responsibility |
| --- | --- |
| [scripts/smoke.py](scripts/smoke.py) | Small baseline pipeline example. |
| [kvtc/codec.py](kvtc/codec.py) | Calibration, compression and reconstruction. |
| [kvtc/pca.py](kvtc/pca.py) | PCA fitting. |
| [kvtc/dp.py](kvtc/dp.py) | Precision allocation. |
| [kvtc/quant.py](kvtc/quant.py) | Quantization and dequantization. |
| [kvtc/serialize.py](kvtc/serialize.py) | Bit packing, payload serialization and entropy coding. |
| [kvtc/cold_store.py](kvtc/cold_store.py) | Page encoding, compact index and selected-page decoding. |
| [experiments/selector.py](experiments/selector.py) | Read ranking coefficients from stored keys. |
| [experiments/reference_heuristics.py](experiments/reference_heuristics.py) | Approximate importance scores, page ranking and halo. |
| [experiments/model_adapter.py](experiments/model_adapter.py) | Qwen cache capture, question queries, recovery and generation. |
| [experiments/reference_task.py](experiments/reference_task.py) | Adapted planted-fact task generation. |
| [experiments/run_rare_facts.py](experiments/run_rare_facts.py) | Frozen inputs and paired evaluation arms. |
| [PROVENANCE.md](PROVENANCE.md) | Origin of the reproduction and adaptations. |
| [reference_implementation_comparison.md](reports/reference_implementation_comparison.md) | Comparison with the separate ICLR-paper implementation. |

Start with the synthetic smoke script, then the codec and cold store. Read the selector and model adapter next, followed by the experiment runner. This follows the data from calibrated features to a model answer.

## 17. Additional work in the separate reference codebase

The attached comparison identified work not fully inventoried in the original guide. Matching files were inspected under:

```text
C:\Users\jaygo\Desktop\DESKTOP\Research Papers\ICLR paper
```

The attachment calls its workspace `KV_project_handoff`. This review verifies the matching local reference files; it does not establish that the two directories have identical contents. The primary project remains `Research Papers\KVTC improvement`.

### 17.1 Implementation inventory and integration status

| Reference addition | What the local code implements | Status in the primary paged project |
| --- | --- | --- |
| Question-query refinement | Short runner supports multiple `--refine` rounds; long runner evaluates one round. New queries rerank pages before generation. | Not integrated. The primary selector still performs one selection pass. |
| Corrected grouped KIVI-style baseline | K quantization per channel within token groups; V quantization within channel groups; recent residual window; K4/V2 and K2/V2 controls; BF16 scale metadata and clamp after conversion. | Separate reference runner exists; not an arm in our primary runner. |
| Other low-precision baselines | INT4 retained subsets, K4/V2 subsets, naive and channel-balanced MiKV-style INT2, and fact-location oracle variants. | Not integrated. These are local approximations, not official implementations. |
| Improved retrieval budget | Reads active-token counts from reference results and includes recovered entries in the estimated comparison budget. | Useful starting point; exact total-memory matching remains pending. |
| 16K/32K pretrained results | Saved Qwen2.5-1.5B rare-fact evaluations, chunked prefill and approximate H2O/SnapKV scoring. | Task/model workflow partly adapted; saved accuracy belongs to the reference simulation. |
| GPU component timing | Warmups, CUDA synchronization and repeated measurements of codec operations, query prepass, scoring, selected-row inverse PCA and sliced-cache decode. | Not ported into a complete indexed-page timing harness. |
| Decode scaling benchmark | Synthetic full/reduced caches over context-length and batch-size sweeps, with cache construction outside the timed region. | A useful microbenchmark; not primary-system end-to-end evidence. |
| `kvtcx` extensions | Weighted DP error, scales shared across token groups, replaceable projection/reconstruction methods and permutation bases. | Experimental reference fork; primary `kvtc` retains its documented standard path. |
| Larger calibration/export CLI | FineWeb-Edu and OpenR1-Math loaders, local-text fallback, configurable token budgets, shared-basis precision reassignment and `basis.pt`/configuration export. | Separate tool; not the primary rare-fact runner's calibration workflow. |

Reference sources:

- [Short two-tier runner](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/exp_r10_twotier.py>) and [long two-tier runner](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/exp_long_twotier.py>).
- [Corrected grouped KIVI runner](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/exp_kivi_fair_long.py>), [INT4/K4V2 runner](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/exp_r10_int4_baseline.py>) and [MiKV-style runner](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/exp_r10_mikv_fair.py>).
- [Latency script](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/exp_t1_latency.py>) and [scaling script](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/exp_t1b_scaling.py>).
- [Experimental codec](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/kvtcx/codec.py>), [experimental DP](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/kvtcx/dp.py>) and [calibration CLI](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/calibrate_kvtc.py>).

These absolute links refer to the local reference checkout. When moving the project to another computer, the relative filenames in the inventory identify the corresponding files; the reference checkout is a separate dependency for reading them.

### 17.2 Refinement: what can be reused and what must change

The reference refinement loop is:

```text
initial selected set
    -> question forward pass over that set
    -> new query vectors
    -> rerank pages
    -> replace recalled set and reunite with hot entries
    -> evaluate another answer
```

This happens before generation, not after each generated token. The reference obtains refinement queries from original cache entries under a mask, including original full-precision entries at newly recalled positions. Final answers use the reconstructed cold cache merged with original hot entries. Therefore the refinement query pass has access to a more accurate cache than the deployed cold-recovery path would provide.

If adapted, refinement must use `active.layers` returned by our `recover`, then call the stored-coefficient selector and real indexed recovery again. It must count repeated query passes, repeated page reads and any retained decoded-page buffers. An all-original masked refinement pass should only be a clearly labeled diagnostic control.

### 17.3 Corrected baseline and budget qualifications

The corrected KIVI runner estimates the two-tier budget as:

\[
B_{\mathrm{comparison}}=b_{\mathrm{cold}}N+N_{\mathrm{active}}(4p).
\]

Here `4p` is two 16-bit cache types per token, and `N_active` is read from the reference run's retained-token counts. This improves on counting only hot entries.

However, `b_cold` is a supplied or default bytes-per-token estimate, not an exact per-document archive byte count retrieved from the JSON. The baseline's per-token quantized-size formula also needs residual-window and partial-group accounting. It produces reconstructed model-precision tensors for inference, rather than a deployed packed low-bit cache. Shared bases, coefficient buffers and temporary tensors remain outside this estimated comparison.

The local [corrected 32K result](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/results_v2/kivi_fair_32k.json>) has **4 recorded documents**, despite a configured target of 40:

| Unconstrained control | Exact-style correct | Reworded correct |
| --- | ---: | ---: |
| Vanilla | 4/4 | 4/4 |
| K4/V2 | 4/4 | 3/4 |
| K2/V2 | 2/4 | 0/4 |

These partial results expose a weak K2/V2 control; they do not establish that our method beats a valid low-precision baseline. Numerical finiteness and near-baseline task behavior must be checked before attributing failure to retention policy.

The attachment cites `results_v2/KIVI_CORRECTED_findings.md`. That file was **not found in the inspected local reference folder**. The table above is derived directly from the available JSON, not from that missing findings document.

### 17.4 Saved quality and timing evidence

The [16K summary](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/results_v2/long_16k_summary.md>) and [32K summary](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/results_v2/long_32k_summary.md>) exist and report 40-document evaluations. The main quality scripts fully decompress the cold cache and mask selected entries; their selectors derive coefficients from original unquantized keys. Those results do not validate our stored-coefficient ranking or independently decoded page path.

The [latency findings](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/results_v2/T1_latency_findings.md>) record approximately 27.2–27.7 ms per decode token across several cache policies at a short context on an RTX 4090 Laptop GPU. This is useful negative evidence: attending to fewer tokens did not produce a material decode-speed benefit in that setup.

The [scaling findings](</C:/Users/jaygo/Desktop/DESKTOP/Research Papers/ICLR paper/results_v2/T1b_crossover_findings.md>) report faster reduced-cache decode at some longer-context/batch settings. Those measurements use synthetic caches and exclude the complete retrieval path. Their crossover and speed ratios are specific to that harness and hardware; they are not H200 predictions, equal-accuracy results, or proof of equal-memory end-to-end speedup.

Selected-row inverse-PCA timing in the reference covers keys, not independent-page entropy decoding plus full K/V recovery and merging. Some reported end-to-end figures combine component measurements rather than timing the full deployed request. The primary paged system still needs isolated, complete-path measurements.

### 17.5 Additional codec research inventory

Matching scripts and all eight listed JSON artifacts were found in the reference folder. Their existence documents experiments, not universally successful optimizations. This review checked their scope and artifact presence; it did not rerun them or independently validate every numerical conclusion.

| Experiment | Reference script | Saved artifact under `artifacts/` | What it investigates |
| --- | --- | --- | --- |
| D1 | `exp_d1_attnweights.py` | `d1_attnweights.json` | Query-derived coordinate weights for precision allocation, including positional considerations. |
| D1 granularity | `exp_d1_granularity.py` | `d1_granularity.json` | Whether DP block granularity changes the effect of weighting. |
| D2 | `exp_d2_blockscale.py` | `d2_blockscale.json` | Sharing quantization scales/shifts across tokens versus the extra quantization error. |
| D3 | `exp_d3_compressed_attn.py` | `d3_compressed_attn.json` | Value aggregation in coefficient space and its costs; not an implemented production attention kernel. |
| D4 | `exp_d4_kv_asymmetry.py` | `d4_kv_asymmetry.json` | K/V transform choices and unequal bit budgets. |
| H1 | `exp_h1_validate.py` | `h1_validate.json` | Unequal K/V budgets checked with teacher-forced continuation KL. |
| R1 | `exp_r1_structured_basis.py` | `r1_structured_basis.json` | Structured alternatives to a dense PCA basis. |
| Entropy headroom | `measure_entropy_headroom.py` | `entropy_headroom.json` | Symbol entropy, layouts and temporal conditional-entropy estimates. |

Weighted DP keeps a basis fixed and weights coordinate distortion. It is not equivalent to learning a new attention-optimal transform, nor does it exactly optimize the complete softmax attention-output error. Likewise, coefficient-domain value aggregation is an algebraic possibility, not evidence that the entire attention operation runs efficiently on entropy-coded bytes.

### 17.6 Calibration/export workflow

`calibrate_kvtc.py` loads FineWeb-Edu and OpenR1-Math when available, supports a local-text path, collects cache features, calibrates a shared basis, reassigns precision at multiple compression ratios, and exports artifacts. This is a useful separate workflow to consider adapting.

Its dataset names, quality-field handling, sampling/balancing and fallback behavior require verification before asserting reproduction of the paper's exact data recipe. Merely having the loaders does not show that saved rare-fact runs used them; those runners use local-text calibration.

### 17.7 Updated adaptation order

1. Preserve the primary indexed archive, global protection and stored-key selector controls.
2. Run the prepared pretrained full/all-page and oracle checks on the H200.
3. Adapt corrected low-precision controls into the frozen-manifest runner and audit their actual memory representation and control quality.
4. Add isolated complete-path timing, reusing useful synchronization and warmup patterns from the reference scripts.
5. Test refinement using reconstructed active entries only, with additional cost reported.
6. Consider calibration export and codec research branches as separate ablations after the basic quality and cost measurements are reliable.

No model implementation, tests, pretrained evaluations or timings were changed or rerun as part of this documentation review. Features listed as separate reference work remain unintegrated unless explicitly stated otherwise above.

## 18. Primary evaluation infrastructure added after the review

The following changes were implemented after the documentation-only review above:

| File | New functionality |
| --- | --- |
| [baselines.py](experiments/baselines.py) | Grouped K2/V2, K4/V2 and K4/V4 round-trip controls; finite-value checks, residual preservation, partial groups and estimated packed byte accounting. |
| [metrics.py](experiments/metrics.py) | English QA F1, normalized exact match, substring success, Wilson intervals, first-token logit KL and synchronized timing helper. |
| [calibration.py](experiments/calibration.py) | Identity-checked, tensor-only reusable calibration artifacts. |
| [prepare_longbench.py](experiments/prepare_longbench.py) | Complete English QA split manifests, pinned revisions/archive hashes, explicit overlength policy and question-blind document prefix. |
| [ablation_suite.py](experiments/ablation_suite.py) | Comparison plans and 16-variant one-factor ablation plans; sequential subprocess execution. |
| [summarize_results.py](experiments/summarize_results.py) | Partial-run safeguards, raw CSV exports and document-paired bootstrap F1 intervals. |
| [preflight.py](experiments/preflight.py) | Larger-model architecture/context checks and calibration memory component estimates without downloading weights. |
| [run_rare_facts.py](experiments/run_rare_facts.py) | Extra baselines, realistic refinement, atomic reporting, full dataset manifests, source/config identities and document-level resume. |
| [model_adapter.py](experiments/model_adapter.py) | Profiled generation with optional EOS stopping and first-answer logits. |
| [test_evaluation_tools.py](tests/test_evaluation_tools.py) | Baseline numerical/storage controls, metrics, dataset integrity, plans, paired statistics and checkpoint identity. |

Refinement now captures queries from `active.layers` returned by the indexed
recovery adapter. It does not access original cold entries. Subsequent recovered
sets still use the shared compressed archive and original positional handling.

The larger-model targets are Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct through
the existing Qwen2 architecture adapter; they have not been run locally. The dataset
scope is six complete English QA splits, not the entire LongBench suite. Prompts are
document-first and differ from official leaderboard prompts. Estimated packed
low-bit sizes are distinct from actual model-precision baseline tensors.

The report includes raw answers, quality metrics, first-token KL, actual archive
and active-cache sizes, selector buffer sizes, original positions, page/token
fractions, phase timings and whole-quality-harness GPU peaks. Timing samples are
not isolated serving benchmarks, and no equal-total-memory claim is made.

The complete local-development, GitHub handoff and H200 commands are in
[EVALUATION_RUNBOOK.md](EVALUATION_RUNBOOK.md). Pretrained accuracy, complete final
ablations, official/validated equal-memory baselines and isolated end-to-end systems
measurements remain outstanding.
