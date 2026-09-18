# Reference implementation review and adaptation — 2026-09-16

## Conclusion

The two projects share the same research design: KVTC compression, contiguous token pages, a hot set, and query-dependent recall. They implement complementary pieces. Keep the primary project's tested indexed archive and numerical fixes; adapt the reference's data generation and model workflow. Do not substitute its full-cache masking path for real selective decompression.

Primary project: `C:\Users\jaygo\Desktop\DESKTOP\Research Papers\KVTC improvement`.
Reference project: `C:\Users\jaygo\Desktop\DESKTOP\Research Papers\ICLR paper`.
The primary codec files matched the working workspace before this adaptation. Both original source families remain available.

## What the reference actually contains

| Component | Reference implementation | Primary project before this adaptation |
| --- | --- | --- |
| KVTC core | Shared cross-layer K/V PCA, quantization, DP, CPU entropy streams | Same lineage, with quantization/DP fixes |
| Storage pages | Logical page masks, typically 16 tokens; full-document serialized cache | Independently decompressible v2 archive, typically 128-token pages |
| Retrieval | Attention-mass ranking over projected keys, top-k plus page halo | Explicit page IDs |
| Model | Qwen/Qwen2.5-1.5B-Instruct, real inference scripts | Synthetic codec tests |
| Model cache | Masks in main quality scripts; separate physical-slicing helpers/timing scripts | Position-preserving decoded K/V rows |
| Dataset | Locally generated rare-fact documents, exact and reworded prompts | Random tensors |
| Results | Saved 1.8K, 16K and 32K task outcomes and timing diagnostics | Verified codec/paging controls |
| Full selective GPU pipeline | Not implemented end-to-end | Not implemented end-to-end |

The reference also contains `kvtcx`, an experimental fork with weighted distortion and multi-token quantization scales, plus multiple negative-result studies. Their broader novelty or impossibility claims were not independently established by this code review.

## Evidence and limits

1. `exp_long_twotier.py:245–248` compresses then fully decompresses the cold cache and constructs full reconstructed model layers. `exp_r10_twotier.py:193–195` does the same. The main two-tier arms use an attention mask over this full state (`exp_long_twotier.py:300–304`). This tests answer quality under a selected subset but does not demonstrate selective entropy decoding or reduced resident memory.
2. `exp_long_twotier.py:255` computes `Kd` from the original unquantized `K0`; `exp_r10_twotier.py:196` likewise uses original projected keys. This is not evidence for retrieval from the stored lossy representation with zero additional memory. `page_mass_scores` also constructs approximate per-head keys for all tokens in its chosen layer band.
3. Refinement queries use original `layers` at newly recalled positions (`exp_long_twotier.py:306`), not the reconstructed recalled cache. Do not carry this precision advantage into a deployment path. The adapted runner currently omits refinement.
4. `exp_t1_latency.py:17–20` explicitly acknowledges that its monolithic entropy streams cannot be selectively decoded. Its partial inverse-PCA measurement reconstructs keys from precomputed coefficients, not the full selected-page K/V recovery pipeline. The separate scaling experiments are useful diagnostics, not proof of end-to-end retrieval speedup.
5. `exp_long_twotier.py:315` budgets cold bytes plus hot bytes, without recalled active entries. Corrected KIVI files exist, but `results_v2/kivi_fair_32k.json` contains only **4/40** documents. Do not treat its percentages as a completed benchmark.
6. H2O/SnapKV approximations are global, not official per-head implementations. Answer success is `target in decoded_text`, not strict equality of the whole output. Retain these labels and scoring rules when making a reproduction comparison.
7. Reworded questions in the long-context script target a different planted fact from the exact question in the same document. This is not a controlled same-fact paraphrase comparison.

## Saved results inspected (not rerun on a pretrained model here)

Both long-context JSON files contain 40 documents. In the 32K file at hot-CR 16:

| Arm | Exact correct / 40 | Reworded correct / 40 |
| --- | ---: | ---: |
| Vanilla | 40 | 37 |
| Full KVTC | 37 | 30 |
| Two-tier k=8 | 38 | 27 |
| Two-tier k=32 | 40 | 33 |

These are recorded outcomes of their original simulation, not results of the primary paged implementation. They should motivate a controlled rerun, not be copied into our results.

## Dataset compatibility

The model work uses a synthetic rare-fact task, not a standard published evaluation dataset. Four paper/proposal texts and Python standard-library source provide filler. Four sentences with random six-digit reference codes are inserted; questions ask for one code. Short-context generation uses approximately 1,800 tokens; long generation supports 16K/32K.

The adapted `experiments/reference_task.py` preserves the reviewed generator functions and prompts, changing only local text-path resolution. `datasets/reference_texts/` contains the four supplied texts, with source hashes recorded in `experiments/reference_sources.json`.

Historical document token IDs are absent from the supplied result files. The same seed does not guarantee the same historical text because Python module sources and tokenizer versions can differ. The new runner freezes calibration/document/question token IDs to a JSON manifest and reuses that exact manifest across comparisons. This reproduces the protocol, but cannot retroactively prove identity with historical examples. Calibration/evaluation filler overlap is preserved and disclosed for reproduction; disjoint evaluation remains future work.

## What was adapted into the primary implementation

- `experiments/model_adapter.py`: chunked prefill, real model-cache capture, shared feature layout, query extraction, physical cache slicing, original RoPE positions, selected-page recovery and deduplicated hot/cold merging. Qwen2 full-attention, batch-one scope is explicit.
- `experiments/reference_task.py`: reference task generation and calibration protocol, with portable local text files.
- `experiments/reference_heuristics.py`: reviewed global attention-score estimator and page-mass heuristic from the supplied scripts.
- `experiments/selector.py`: coefficient scan reads the actual stored quantized key streams. It does not read the original unquantized keys or decompress value streams. It does scan all key pages and allocate an N-by-topk FP32 buffer; those costs are reported. This is a selector baseline, not an efficient zero-overhead retrieval claim.
- `experiments/run_rare_facts.py`: seven arms: vanilla, monolithic full KVTC, paged full KVTC, hot only, oracle-selected pages, random pages, and stored-coefficient-selected pages. All recall arms call the real indexed decoder and physically assemble a smaller attention cache. Full decoding occurs separately for diagnostic arms and their memory remains resident during this quality harness, so its peak memory/timing must not be presented as deployment measurements.
- `scripts/smoke_model_integration.py`: offline, randomly initialized tiny Qwen2 control exercising the entire runner without pretrained downloads.

The runner uses an explicit hot-token budget, not a claim of byte-matched hot-CR budgets. It preserves the global protected region even if this expands that requested budget and reports the actual count. DP stride 16 is explicitly recorded to match the reference search restriction; our default codec remains stride 1. Page size defaults to 128, while 16 is available for a direct reference-style comparison with measured storage overhead.

## Local verification

- Reference project's original CPU test suite: **32 passed**.
- Primary suite after adaptation, including the tiny-model controls: **73 passed**.
- Full seven-arm runner completed on a tiny random-weight Qwen2 model. Its accuracy is meaningless and is explicitly labeled as such.
- Tests cover all-page model-logit agreement with full decode, sparse versus masked native-cache agreement using original positions, hot-entry precedence and duplicate removal, selected-page-only recovery, actual stored coefficient decoding, reproducible planted facts and manifest reuse.

No pretrained Qwen quality run or H200 timing run has been executed in this session. The installed local environment is CPU-only. See `H200_RUNBOOK.md` for the next runs.

## Remaining work toward the final target

1. Run the prepared pretrained Qwen experiment on the H200, using a single frozen manifest. Establish full/all-page and oracle controls before interpreting the automatic selector.
2. Compare 16/64/128-token pages at matched decoded-token budgets, not just the same top-k count. Measure whether quantized-coefficient ranking preserves page recall and answers.
3. Isolate each arm's allocations and include cold archive, shared basis, hot entries, recalled entries, selector state and transient peak buffers. Port/fix a complete KIVI baseline before equal-memory claims.
4. Implement GPU-resident storage/entropy decoding or state clearly that the system uses CPU storage. Current zlib pages are CPU bytes; moving matrix multiplication to CUDA does not make the archive GPU-resident.
5. Measure the complete timed path, including query pre-pass, key scan, transfers, entropy decode, reconstruction and sparse attention. Then optimize scan cost and avoid redundant reconstruction of hot entries within selected pages.
6. Expand beyond the synthetic task, remove calibration overlap, and evaluate additional models/seeds. No novelty or publication-readiness conclusion is established here.
