# H200 phase-one readiness analysis — 2026-09-28

## Decision

The CPU-tested orchestration and numerical controls are ready for a native GPU
smoke run. The production benchmark is **not ready to start on the reported H200
shell yet**. Resolve the four external/runtime blockers below and require the GPU
smoke matrix to finish before starting the full dataset.

## Evidence completed on the development host

- Main suite: **158 passed, 26 skipped** in 48.83 seconds. The skipped group
  contains environment-specific/native checks that are covered separately or
  require CUDA.
- Isolated Transformers 4.45.2 adapter suite: **25 passed** in 13.39 seconds.
- Python compilation: all files under `experiments`, `kvtc`, `scripts`, and
  `tests` compiled successfully.
- Dependency consistency: `pip check` passed in the main and baseline-test
  environments.
- Synthetic codec smoke: completed serialization/reconstruction with finite
  results; this is not an accuracy or performance result.
- Tiny random-weight Qwen2 integration: all 14 wiring arms completed and the
  second invocation resumed without duplicate rows; its accuracy is meaningless.
- Complete CPU orchestration rehearsal: all **64 model/benchmark/method jobs**
  completed in two scheduler passes. It seeded a partial checkpoint, resumed it,
  revisited completed checkpoints, and generated combined JSON/CSV/Markdown
  reports. Score fields remained empty by design.

The rehearsal exposed intermittent Windows `os.replace` permission failures from
short-lived file locks. Atomic result writes now retry bounded transient
`PermissionError`s and a regression test covers the behavior.

## Blockers observed on the H200 shell

1. `nvcc` reports CUDA toolkit **13.1**. The baseline environment pins PyTorch
   2.5.1 `cu124` and builds FlashAttention 2.6.3 from source. Select a CUDA 12.x
   compiler, with 12.4 as the pinned target, before rerunning setup. The NVIDIA
   driver may remain at its current version.
2. Only **one H200 was visible** when preflight ran, while the requested matrix
   and 72B model plan require a five-GPU allocation. Exporting five IDs cannot
   expose GPUs that the scheduler/container did not allocate.
3. Setup stopped before creating `phase-ours`, `phase-judge`, and the pinned
   upstream checkouts; the partial `phase-baselines` environment also lacks
   FlashAttention. Rerun the setup after selecting CUDA 12.4.
4. The authenticated Hugging Face account returned 403 for
   `meta-llama/Llama-3.1-8B-Instruct`. Accept the model terms/get approval with
   that account, or the required four-model matrix cannot be complete.

Disk (~467 GiB free) and host RAM (~2 TiB) passed the configured thresholds.

## Code-path assessment

- The planner keeps every requested model/method/benchmark cell visible, rejects
  duplicate or unsupported cells, and requires 64 executable jobs when the
  complete-plan gate is used.
- The scheduler reserves disjoint visible GPU IDs, permits independent jobs to
  share a wave, and allocates multiple GPUs to 72B. This is process scheduling
  plus Hugging Face layer sharding, not tensor parallelism.
- Checkpoints commit after every example with input/source/environment identity.
  Resume skips committed examples and rejects changed inputs. An interrupted
  example restarts from its stable per-example seed; token-level KV state is not
  checkpointed.
- Smoke scores are suppressed, incomplete results are not aggregated, and stale
  LongGenBench judge files cannot be merged.
- The preflight now rejects a CUDA 13.x compiler for the pinned `cu124` baseline,
  captures child-process errors without dumping redundant tracebacks, and checks
  BF16 support on each selected device.

## Research limitations that remain

- CPU tests do not validate FlashAttention, FlashInfer, native baseline kernels,
  BF16 behavior, H200 memory use, or multi-GPU dispatch. Only the real H200 smoke
  can establish those properties.
- Phase one produces quality results. Per-example elapsed time is diagnostic and
  cannot populate the paper's latency, TTFT, peak-memory, bandwidth, or throughput
  table.
- The current `ours` path is a diagnostic CPU-zlib prompt archive with split key
  head/tail streams. Selection reads only key heads and recalls selected tails,
  but it is not an optimized GPU serving implementation or a matched-byte system
  benchmark.
- `ours` retains generated KV exactly and is not yet a fixed-budget long-generation
  implementation. Short LongGenBench prompts may never create an archive; reports
  expose the count through `prompt_archive_samples`.
- Qwen2.5 7B/14B/72B released configs are 32K. The FreeKV 120K protocol used here
  records unscaled extrapolation and does not enable YaRN, so results above 32K
  need that qualification.
- H2O is the KVCache-Factory prefill-compression implementation. The methods use
  published token settings but are not yet matched by total stored bytes.

## Required execution order

1. Select CUDA 12.4 and verify `nvcc --version`.
2. Run the dependency-free CPU rehearsal.
3. Run `scripts/setup_phase_one.sh` to completion.
4. Authenticate an account with Llama access.
5. Obtain five visible H200s and pass online preflight.
6. Run all CPU/integration tests.
7. Run the one-GPU Qwen 7B native smoke.
8. Run and inspect the complete 64-job native smoke.
9. Start production only if all 64 smoke rows are completed.

Exact copy-paste commands are in `H200_COMMANDS.md`.
The clean-clone CPU-only workflow is in `CPU_TESTING.md`.
