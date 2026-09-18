# Unified H200 comparison and resume

This workflow uses the final **Llama-3.1-8B-Instruct**, **Qwen2.5-7B-Instruct** and
**Qwen2.5-14B-Instruct** models. It prepares real runs; the numbers in the supplied image
are illustrative and are never copied into reports. Local verification uses
random tiny models and CPU tests. H200 kernels, pretrained quality and model
sharding remain unvalidated until run on the GPU machine.

## What is implemented

- A common plan/launcher across isolated environments, `--gpus N`, conservative
  memory screening, per-job logs, continuation after failures, and combined JSON,
  CSV and Markdown reports.
- Official KIVI Llama bridge, and an ArkVale bridge for upstream-compatible Llama
  configurations. Both upstream sources are pinned in
  `configs/upstream_baselines.json`; fetch/build separately from the modern stack.
- The local KVTC adapter now accepts Llama as well as Qwen2. Its sparse-position,
  reconstruction, and selective-decoding tests run against both tiny architectures.
- GPU-only balanced model placement for the full-cache worker and KVTC quality
  runner. No implicit weight quantization or CPU/disk model offload. GPU-specific
  cache reconstruction and cross-device synchronization are implemented; multi-GPU
  execution still requires validation on actual hardware.
- LongBench's 16 English/code tasks with the published task-specific scorers in
  the pinned KIVI checkout. Complete supplied splits are preserved. Summarization
  uses ROUGE-L, classification/retrieval use their scorers, code uses LongBench
  code similarity (not mislabeled BLEU). Prompts are task templates split around
  the context, without model-specific chat wrapping; these are not leaderboard
  reproductions.
- Frozen local RULER/MATH-500 exports. RULER uses explicitly labeled reference
  string recall, not effective-context-length aggregation. MATH uses final boxed
  string EM, not symbolic equivalence. No synthesized GSM8K-Long protocol.
- Page sweeps 16/32/64/128 with 1,024 recalled tokens and zero halo. This holds
  requested recall tokens fixed, not total GPU memory or total attention tokens.

## Important compatibility gaps

| Component | Llama-3.1-8B | Qwen2.5-7B / Qwen2.5-14B |
|---|---|---|
| Full cache SDPA/FA2 | GPU worker implemented | GPU worker implemented; sharding available when needed |
| Our KVTC quality harness | Tiny-model checks passed | Tiny Qwen2 checks passed; pretrained runs pending |
| Official KIVI | Llama bridge implemented; H200 validation pending | No Qwen2 upstream adapter |
| Original ArkVale | Unsupported custom RoPE for llama3 | Unsupported architecture |
| Quest, RocketKV, ShadowKV, FreeKV | Registry / adapter pending | Registry / adapter pending |
| LouisKV, HBW-KV | Paper identified; official source unverified | Same |
| Official SnapKV, H2O | Pending; local approximations retain `_approx` labels | Same |
| FA3 | Not integrated; FA2 is never labeled FA3 | Same |

Unmodified ArkVale pins Transformers 4.40.0, uses a custom scalar RoPE kernel,
and compiles page sizes 16/32, head dimension 128, GQA groups 1/4/8. It must be
ported and checked before claiming results on the requested models. Do not
remove the compatibility rejection to force a run. KIVI pins Transformers
4.43.1; the current project uses 5.16.1. Separate processes prevent import conflicts.

Both Qwen models' released configs have `max_position_embeddings=32768`. Their model cards
describe an explicit YaRN setup for longer contexts; the current sparse RoPE
inverse rejects YaRN. Llama's 131,072 limit includes the answer budget: a 131,072
token prompt plus 128 generated tokens exceeds it. 256K is outside all validated
configurations. Reduce prompt length to reserve output tokens, without silently
changing the benchmark's definition of context length.

The new baseline taxonomy is in `configs/baseline_registry.json`. RocketKV is
prefill eviction plus sparse attention, whereas ShadowKV is low-rank CPU offload.
PG-19/WikiText-103 PPL, GSM8K-Long, L-Eval/BLEU, head/layer budget allocation, and
FP16/INT8 index ablations are still implementation work; they are not inferred
from QA metrics. `--topk` already provides a low-rank selector-coordinate sweep.

## 1. Set up the H200 environments

Copy the source project, including `configs`, `scripts`, and the calibration
texts, to Linux. `external/` and environments are intentionally ignored by Git.
The fetch script creates exact pinned checkouts and initializes ArkVale submodules.
Do not copy Windows `.venv` or compiled extensions to Linux.

```bash
python -m scripts.fetch_baselines
# In your modern CUDA PyTorch environment:
python -m pip install -r requirements-model.txt -r requirements-evaluation.txt
python -m pip install flash-attn --no-build-isolation

# Separate Python 3.10 / CUDA 12.x development environments:
bash scripts/setup_h200_baselines.sh kivi
# Optional: original ArkVale cannot yet evaluate our two selected architectures.
bash scripts/setup_h200_baselines.sh arkvale

python -m pytest -q
nvidia-smi
```

The setup script builds Hopper (`sm_90`) KIVI kernels; ArkVale's CMake uses the
visible GPU's native architecture. Build scripts have not been executed locally
because this machine has CPU PyTorch and no CUDA compiler. Runtime/toolchain
compatibility must be established on the H200. Save pip freezes and `nvidia-smi`.

Copy `configs/h200_environments.example.json` to a new file and replace each path
with the actual Linux Python executable. Include only installed environments.
The runner fails clearly if the upstream Transformers version does not match.

## 2. Freeze each model's inputs once

Resolve an immutable model commit first. Gated Llama access requires an already
authorized Hugging Face login on the host. A small read-only metadata query is:

```bash
python -c "from huggingface_hub import model_info; print(model_info('meta-llama/Llama-3.1-8B-Instruct').sha)"
python -c "from huggingface_hub import model_info; print(model_info('Qwen/Qwen2.5-7B-Instruct').sha)"
python -c "from huggingface_hub import model_info; print(model_info('Qwen/Qwen2.5-14B-Instruct').sha)"
```

Use each model's returned hash for `$LLAMA_REV`, `$QWEN7_REV` and `$QWEN14_REV`; do not use `main`.
Use a downloaded official LongBench `data.zip`, whose bytes are hashed:

```bash
python -m experiments.prepare_longbench --suite16 \
  --model meta-llama/Llama-3.1-8B-Instruct --revision "$LLAMA_REV" \
  --data-zip /data/LongBench/data.zip --max-context 131072 \
  --out-dir outputs/manifests/llama31

python -m experiments.prepare_longbench --suite16 \
  --model Qwen/Qwen2.5-7B-Instruct --revision "$QWEN7_REV" \
  --data-zip /data/LongBench/data.zip --max-context 32768 \
  --truncate-middle --out-dir outputs/manifests/qwen7

python -m experiments.prepare_longbench --suite16 \
  --model Qwen/Qwen2.5-14B-Instruct --revision "$QWEN14_REV" \
  --data-zip /data/LongBench/data.zip --max-context 32768 \
  --truncate-middle --out-dir outputs/manifests/qwen14
```

The Qwen commands explicitly truncate overlength contexts; every affected row and
the policy are recorded. Their scores must not be called untruncated 128K results.
Prepare small pilot manifests separately before complete splits; never alter a
frozen manifest to resume a different dataset. Calibration uses the existing local
text protocol and has not been audited for overlap with all new tasks.

For a local RULER generator export (`input`, `outputs` fields) or MATH-500 export
(`problem`, `answer` fields), use:

```bash
python -m experiments.prepare_benchmark --benchmark ruler \
  --input-jsonl /data/ruler/16k/niah_single_1/validation.jsonl \
  --source-revision 'ACTUAL_GENERATOR_COMMIT_AND_CONFIG' \
  --model meta-llama/Llama-3.1-8B-Instruct --revision "$LLAMA_REV" \
  --max-context 32768 --new-tokens 128 --out outputs/manifests/ruler16k.json
```

Use `--benchmark math500 --new-tokens 2048` for mathematical reasoning. The
importer keeps all supplied rows, verifies token budgets, and records input hashes.
It does not claim that an arbitrary export contains the full official split.

## 3. Select GPU count and create the plan

```bash
python -m experiments.h200_suite \
  --manifests outputs/manifests/llama31/*.json outputs/manifests/qwen7/*.json outputs/manifests/qwen14/*.json \
  --environments configs/my_h200_environments.json \
  --gpus 2 --gpu-memory-gib 130 \
  --methods ours,full,kivi,arkvale,quest,rocketkv,shadowkv,freekv,louiskv,snapkv,hbw_kv,h2o \
  --context 65536 --batch-size 8 \
  --out-dir outputs/h200 --plan outputs/h200/plan.json
```

Use `--gpus 1`, `2`, `4`, etc. according to your allocation. If a cluster scheduler
sets `CUDA_VISIBLE_DEVICES`, the launcher respects those IDs. The memory argument
is usable capacity in GiB **per GPU**, from the host; it defaults to 130.

- Small quality jobs get independent GPUs when possible.
- Models that do not fit one GPU use balanced layer sharding for full cache and
  our quality harness. This is model parallelism, not tensor-parallel serving.
- System jobs run one at a time to avoid concurrent jobs perturbing timings.
- Official ArkVale/KIVI currently require a single GPU. A multi-GPU allocation
  does not manufacture architecture or model-sharding support.
- Insufficient-memory/unsupported cells remain in the plan with reasons. No model
  weights are downloaded to make a plan. Estimates include weights, full KV,
  reserve, and extra diagnostic caches for ours, but are not an OOM guarantee.

The suite uses FP16, randomized rank-1024 calibration, top-256 selector coordinates,
1,024 hot tokens, and no halo. These are explicit pilot settings, not claims of
optimal rank or a 12.5% total-memory match. Review the plan before expensive runs.
Use `--ablations --kind quality` for page16/32/64/128. Prepare separate context
manifests for 16K/32K/64K/128K sweeps. Unsupported head/layer allocation and index
precision variants are not generated as fake runnable flags.

## 4. Execute and resume

```bash
python -u -m experiments.h200_suite --plan outputs/h200/plan.json --execute
# After interruption: exactly the same command.
python -u -m experiments.h200_suite --plan outputs/h200/plan.json --execute

# Regenerate tables while jobs are unfinished or after copying results back:
python -m experiments.comparison_report --plan outputs/h200/plan.json \
  --out-dir outputs/h200/comparison
```

Do not start two copies of the same plan concurrently. A job failure is recorded,
its log is kept next to `result.json`, and other jobs continue. Use a **new plan
and output directory** to change GPU allocation, source, model revision, manifest,
or benchmark settings. Reusing a complete checkpoint with a changed identity fails.
An interrupted upstream example/repeat restarts only that unit; completed units
are saved by fsync plus atomic rename. Ours commits whole documents and reuses
its identity-checked calibration artifact. It does not checkpoint partial GPU KV
state. Concurrent ours jobs are serialized to avoid calibration-file write races.

The combined reports show scores only for complete jobs. Missing/unsupported
results stay N/A. Timing reports contain decode ms/token, TTFT, throughput,
allocator peaks, actual batch/context and GPU count. TTFT includes full prefill,
question ingestion and first-token selection, excluding model loading/tokenization.
System workloads repeat one real tokenized document to exact length and replicate
it across batch rows; they are synthetic load tests. Memory includes model/cache
pools and prefill/decode peaks, summed across devices, and excludes allocations
outside PyTorch. Bandwidth utilization is N/A until measured with hardware DRAM
counters; GPU utilization is not substituted for bandwidth.

The official bridges ingest question tokens one at a time; our harness processes
the question as a chunk. Keep the full-cache controls from each execution path.
Do not pool unequal prompt protocols or treat KIVI's 2-bit payload as exactly
12.5% of total cache memory: scales, residual tokens, padding and metadata count.

## Primary sources

- [ArkVale source and API](https://github.com/pku-liang/ArkVale)
- [KIVI source and GQA/Llama support](https://github.com/jy-yuan/KIVI)
- [Qwen2.5-7B model card and long-context configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen2.5-14B model card and long-context configuration](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- [Quest](https://github.com/mit-han-lab/Quest), [RocketKV](https://github.com/NVlabs/RocketKV),
  [ShadowKV](https://github.com/ByteDance-Seed/ShadowKV), [FreeKV](https://github.com/sjtu-zhao-lab/FreeKV)
- [LouisKV paper](https://arxiv.org/abs/2510.11292), [HBW-KV paper](https://openreview.net/pdf?id=sQjYtFSEuZ)
