# H200 comparison implementation status — 2026-09-18

The final models are Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct and
Qwen2.5-14B-Instruct, recorded in `configs/h200_models.json`. The comparison
launcher takes `--gpus N`, screens memory, allocates disjoint devices to independent
quality jobs, and uses GPU-only layer sharding where implemented. System jobs run
alone. Plans preserve unsupported entries with reasons rather than silently
substituting models, quantizing weights, or extending RoPE.

Both original repositories were fetched and inspected. ArkVale is pinned to
`0c3b45ebe7ecc8c81ff4bc0f9c68c04a0866de88`; KIVI is pinned to
`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`. Original source is kept in ignored
`external/` checkouts and is not modified. Reproducible fetch/build instructions
are in `H200_COMPARISON_RUNBOOK.md` and `scripts/setup_h200_baselines.sh`.

KIVI's official Llama bridge can be attempted on Llama-3.1 in a separate
Transformers 4.43.1 environment. The bridge enables KIVI's legacy flash-attention
mask flag, uses its packed quantized kernels, and slices unused prefill logits at
the LM head to avoid a context-by-vocabulary allocation. These bridge choices are
not changes to the attention/quantization algorithm, but need H200 validation.
ArkVale's bridge is implemented for its original compatible Llama configurations;
it is deliberately excluded for Llama-3.1's unsupported custom RoPE and Qwen2.

LongBench-16 preparation/scoring and RULER/MATH export preparation are added.
Combined tables keep tasks, metric names, model revisions, manifest hashes and
configuration hashes distinct. The local approximate H2O/Snap/low-bit arms retain
their original labels. Bandwidth, PPL and missing official methods are not assigned
invented scores. The broader baseline list is an implementation registry, not a
claim that every baseline has been integrated.

Atomic upstream checkpoints commit examples/repeats. Existing KVTC checkpoints
commit documents and reuse calibration; finished pinned runs can skip model load.
Resume rejects changed source/settings/data, invalid or duplicated work IDs, and
claims of completion with missing work. Unit tests also exercise torn temporary
files, GPU allocation, failure continuation and incomplete-report handling.

The final CPU suite passed **102 tests**; Python compilation checks also passed.
CPU checks include tiny Qwen, Llama, and Llama-3.1-style RoPE cache reconstruction,
sparse masking and selective decode. The 14-arm random-weight end-to-end smoke and
resume check passed. **No pretrained H200 result, CUDA extension build, or actual
multi-GPU execution has been validated here.** Full details and outstanding
protocol/adapter work are listed in the runbook.
