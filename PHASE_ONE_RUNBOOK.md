# First H200 run: LongBench v2 + LongGenBench

For the complete sequence starting with `git clone`, including monitoring and
runtime estimates, use [GETTING_STARTED_H200.md](GETTING_STARTED_H200.md).

This is the current run plan, superseding the earlier LongBench v1 / 7B–14B plan.
Run these commands from the project directory on the **Linux H200 machine**.
The local Windows checks do not establish H200 performance or multi-GPU correctness.

Models: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`,
`Qwen/Qwen2.5-14B-Instruct`, and `Qwen/Qwen2.5-72B-Instruct`.
Methods: ours, full cache, H2O, SnapKV, Quest, ArkVale, RocketKV, FreeKV.
RocketKV is the selected member of the requested RocketKV/ShadowKV alternative;
ShadowKV is not run in this phase. The complete matrix contains 64 generation jobs.

## 1. Install and authenticate

Prerequisites: Python 3.11, Git, C++ build tools, a CUDA 12.4 toolkit (`nvcc`),
and an NVIDIA driver compatible with CUDA 12.8 runtimes. Installation downloads
dependencies; first execution downloads model weights and datasets. Allow several
hundred GB of disk space. The current ours diagnostic path also needs substantial
host RAM at 120K, especially for 72B; GPU planning does not estimate host RAM.

Copy the entire project, including `configs`, `experiments`, `kvtc`, `scripts`, and
requirements files. Upstream checkouts are fetched at pinned revisions by setup.
Existing checkouts at different revisions or with tracked edits are rejected;
they are not overwritten.

```bash
bash scripts/setup_phase_one.sh
.envs/phase-baselines/bin/huggingface-cli login
nvidia-smi
```

The HF account needs access to Meta's Llama model. Dependencies are isolated:
baseline math uses Transformers 4.45.2; ours uses 5.16.1; judging uses vLLM 0.10.2
and Transformers 4.55.2. Keep `outputs/environment-locks/` with the experiment.

## 2. Smoke-test before committing to the full suite

Optionally repeat the local regression and adapter math tests first:

```bash
bash scripts/test_phase_one.sh
```

Then validate the real GPU execution path:

```bash
bash scripts/run_phase_one.sh --gpus 5 --smoke --out outputs/phase-one-smoke
```

This checks all four models and all eight methods on two examples per benchmark,
with a 4,096-token raw prompt cap and eight output tokens. It includes 72B model
sharding. Each job writes `result.json` and `result.log`. Failures do not stop
other jobs; the command exits nonzero if any job fails. Do not interpret smoke
outputs as benchmark scores: aggregate accuracy is suppressed in smoke reports.

For a quicker first check, use a separate directory:

```bash
bash scripts/run_phase_one.sh --gpus 1 --smoke \
  --models Qwen/Qwen2.5-7B-Instruct --out outputs/phase-one-smoke-7b
```

The native CUDA builds and multi-GPU bridges have not been exercised on this
Windows CPU machine. Examine smoke logs before starting production runs.

## 3. Full run and automatic resume

```bash
bash scripts/run_phase_one.sh --gpus 5 --out outputs/phase-one
```

**Repeat the identical command after an interruption.** Completed examples are
skipped. An interrupted example starts again, with the same per-example sampling
seed; mid-token model/KV state is not checkpointed. LongGenBench contains 400
prompts with up to 16,000 generated tokens each, for every model/method pair, so
the full run is substantial. Completed jobs skip loading their model weights.

The first run freezes model, dataset, calibration-corpus, and judge revisions in
`revisions.json`; freezes identical chat-token inputs for every method under
`manifests/`; and writes the GPU allocation/commands to `plan.json`.
Resume refuses incompatible inputs, source, packages, or per-job GPU layouts.
Keep the source and environments unchanged until the run is finished. To change
an experimental setting, use a new output directory.

`--gpus N` controls the number of visible GPUs the scheduler can reserve. GPU
memory is detected automatically; `--gpu-memory-gib 130` can lower the planning
limit. No automatic weight quantization or CPU/disk model offload is used. Small
jobs run on separate GPUs; 72B uses multiple GPUs through layer sharding. This is
not tensor parallelism and does not require the GPU count to divide the number
of attention heads. The count is estimated from weights, dense KV, and workspace,
with a 10% reserve; it is not a guarantee against OOM.

Select an allocation explicitly, if needed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash scripts/run_phase_one.sh --gpus 5
```

A process lock prevents two launchers or a judge from writing to the same output
directory simultaneously. The lock releases on process termination. Different
output directories are independent: do not start them on overlapping GPUs.

## 4. Judge LongGenBench and collect the combined table

After generation completes:

```bash
CUDA_VISIBLE_DEVICES=0 .envs/phase-judge/bin/python -m experiments.phase_one_judge \
  --out outputs/phase-one --gpus 1
.envs/phase-baselines/bin/python -m experiments.phase_one report --out outputs/phase-one
```

Repeat the judge command to resume. Each individual check is committed atomically.
The judge is the pinned **Qwen/Qwen3-32B** model used by FreeKV's evaluator,
with thinking disabled, temperature 0.95, top-p 0.95, max 50 tokens, and seed 42.
It occupies a separate stage; it does not compete with generation jobs for GPU 0.
The original evaluator's machine-specific model path is replaced by the model ID.

Open `outputs/phase-one/comparison.md` or `comparison.csv` / `comparison.json`.
They include LongBench v2 accuracy (%), LongGenBench completion (%), and judged
once/range/periodic/average accuracy (0–1). Judged accuracy stays empty until the
matching generation has been judged. Failed, incomplete, unsupported, and smoke
jobs do not receive full-run aggregate accuracy. Raw answers, token IDs, source
revisions, implementation labels, and run identities remain with each result.

## Protocol and implementation details that matter

The benchmark source is [FreeKV's accuracy suite](https://github.com/sjtu-zhao-lab/FreeKV/tree/2c8a7d25c9f3c7c15ce15b2f84cd03f477bd7469/accuracy).
LongBench **v2** uses `THUDM/LongBench-v2`, train split, zero-shot multiple choice,
the supplied prompt and exact answer regex, greedy decoding, and 128 output tokens.
It is not the old 16-task LongBench v1 average. LongGenBench uses FreeKV's bundled
`Dataset_short.json`, 400 examples, 16,000 output tokens, temperature 0.95, top-p 1,
the original prefix/block parsing and stop-sign tokenization. Missing blocks are
omitted from FreeKV's judged-accuracy denominator; completion is reported separately.

Prompts are truncated in the middle **before** chat templating at 120,000 tokens,
as in FreeKV. Qwen gets FreeKV's explicit system message. The Qwen models' released
configuration is 32K: this phase deliberately follows FreeKV's longer unscaled
context protocol and records that extrapolation. It does not enable YaRN or claim
that this is the model publisher's recommended 128K setup. All methods receive
the exact same frozen tokens. Ours calibrates on a separate pinned WikiText train
sample, not evaluation questions/answers.

FreeKV/Quest/ArkVale use sink/recent/budget = 128/128/1792 on LongBench v2 and
512/512/1024 on LongGenBench; page size 32; first layer exempt. FreeKV uses two-step
speculation with correction thresholds 0.8/0.9 and avgSM GQA; Quest uses maxS;
ArkVale uses avgS. H2O/SnapKV's prompt capacity is 2,048 tokens with a 32-token
observation window; GQA scores are averaged at KV-head granularity. RocketKV uses
its upstream two-stage capacity formula with the same nominal 2,048-token budget.
These are token settings, **not a matched total-memory claim**.

| Result label | Implementation / qualification |
|---|---|
| Full cache | FreeKV BF16/FlashAttention-2 accuracy path, with a local fix returning the exact KV cache; logits checked against unmodified HF on tiny Llama, Llama-3 RoPE, and Qwen models. |
| Quest / ArkVale / FreeKV | FreeKV's published accuracy implementations; not their original optimized serving kernels. Device-local counters and fresh speculative-query buffers support independent samples. |
| H2O | KVCache-Factory's **prefill compression** variant, not a claim to the original token-by-token H2O oracle. Chunked scoring preserves the dense score definition without allocating an N-by-N matrix. |
| SnapKV | KVCache-Factory prefill implementation, with a local Qwen attention-interface bridge. Generated cache entries are retained. |
| RocketKV | Upstream HF attention math, with local fixes for its Qwen class/rotary interface and device-local rotary tables. Tiny Llama/Qwen decode and reset checks pass; H200 sharding still needs the smoke run. |
| Ours | Existing KVTC prompt archive, hot-cache selection and query scan, applied to the last 64 prompt tokens. The newly generated cache is retained exactly. **This is not yet a fixed-budget long-generation implementation**; short LongGenBench prompts may not use an archive at all. `prompt_archive_samples` records this in the report. |

Generation sampling uses a stable per-example seed and CPU multinomial to make
resume independent of scheduling. This differs from upstream's single advancing
GPU RNG stream, so sampled text is not expected to reproduce FreeKV byte for byte.
The wrapper also stops on a first-token EOS, handles full-cache updater=None,
keeps empty responses as scored examples, and avoids the upstream sample-slicing
and hard-coded output-path issues. Source repositories are not edited.

This first phase produces quality results. Per-example elapsed time is diagnostic;
it is not a serving benchmark. Throughput, TTFT, peak memory, bandwidth utilization,
and the earlier ablation table still require isolated H200 performance experiments.
No numbers from the supplied illustrative figures are inserted as measured results.
