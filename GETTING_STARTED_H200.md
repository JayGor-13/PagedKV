# PagedKV: from GitHub clone to H200 results

Run this guide on the **Linux H200 server**, in order, from the same project
directory. The default is five GPUs. It covers setup, unit/integration tests,
native GPU smoke tests, the full benchmark, judging, monitoring and resume.

The current suite runs Llama-3.1-8B-Instruct and Qwen2.5-7B/14B/72B-Instruct on
LongBench v2 and FreeKV's LongGenBench short dataset. Each model runs ours, full
cache, H2O, SnapKV, Quest, ArkVale, RocketKV and FreeKV: **64 generation jobs**.
RocketKV is selected for the RocketKV/ShadowKV alternative.

## 1. Clone the repository

Choose a storage location with room for model downloads, environments and outputs.

```bash
git clone https://github.com/JayGor-13/PagedKV.git
cd PagedKV
git rev-parse HEAD
```

If already cloned, run `git pull --ff-only` **before starting a new experiment**.
Do not update code or reinstall environments while resuming an existing run:
resume checks source and environment identities.

## 2. Check the server and keep the terminal session alive

The server needs Python 3.11 with `venv`, Git, C++ build tools, a CUDA 12.4 toolkit
for compiling FlashAttention, and an NVIDIA driver supporting CUDA 12.8 runtimes.
If these are missing, have the server administrator provide them before setup.
The CUDA version displayed by `nvidia-smi` is driver capability; `nvcc --version`
reports the installed toolkit. They need not display the same version.

```bash
python3.11 --version
git --version
g++ --version
nvcc --version
nvidia-smi
df -h .
free -h
```

Allow several hundred GB of free disk space. The four BF16 generation models plus
the Qwen3-32B judge represent roughly 270 GB of weights before environments,
datasets, cached revisions and outputs. Ours also has substantial host-RAM use
for long prompts; fitting GPU memory alone does not establish that a run will fit.

If using SSH and `tmux` is installed, start a persistent terminal:

```bash
tmux new -s pagedkv
```

The new session normally inherits the project directory; check `pwd`. Detach with
**Ctrl+B**, then **D**. Reconnect with `tmux attach -t pagedkv`. Without a persistent
session, keep the terminal connected for the duration of a run.

Select the five GPUs allocated to this experiment, adjusting IDs if necessary:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
mkdir -p outputs/logs
set -o pipefail
```

`pipefail` makes failures visible even when output is piped through `tee`.
Do not launch independent runs on the same GPUs simultaneously.

## 3. Install the three isolated environments

```bash
time bash scripts/setup_phase_one.sh 2>&1 | tee outputs/logs/setup.log
```

This installs `.envs/phase-baselines`, `.envs/phase-ours` and `.envs/phase-judge`,
fetches the pinned upstream repositories, and records installed packages under
`outputs/environment-locks/`. The baseline and ours environments use different
Transformers versions; do not combine them into one environment. There is no need
to activate a venv manually: the scripts select the correct Python executable.

Stop and resolve any setup error before proceeding. Setup completion confirms
installation, not native model correctness. On a clean clone, run setup before
the tests: some tests read the pinned upstream benchmark and evaluator files.

## 4. Log in to Hugging Face

The account must have access to `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face.
Accept the model's access terms using that account if needed, then:

```bash
.envs/phase-baselines/bin/huggingface-cli login
.envs/phase-baselines/bin/huggingface-cli whoami
```

The environments share the account's standard Hugging Face cache. Model weights
and datasets download on first use. Initial runs therefore include download time;
subsequent runs generally reuse those files. Do not put access tokens in scripts.

## 5. Run all local regression and adapter tests

```bash
time bash scripts/test_phase_one.sh 2>&1 | tee outputs/logs/tests.log
```

The development-machine result was **115 main tests passed**, followed by
**18 baseline adapter tests passed** in the separate environment. The first test
invocation skips those 18 version-specific tests; the second invocation runs them.
The script also verifies pinned upstream commits and source integrity.

These tests check cache/position handling, adapter math, resumability, scoring,
reporting and scheduling. Their baseline math tests replace CUDA attention with
CPU SDPA. They do **not** replace the native GPU smoke tests below. If any test
fails, inspect `outputs/logs/tests.log` and resolve it before a full run.

## 6. Run a quick GPU smoke test

Start with Qwen2.5-7B to check the installed CUDA stack and all eight methods:

```bash
time bash scripts/run_phase_one.sh --gpus 1 --smoke \
  --models Qwen/Qwen2.5-7B-Instruct \
  --out outputs/phase-one-smoke-7b \
  2>&1 | tee outputs/logs/smoke-7b.log
```

This uses one selected GPU and a separate output directory. With `--smoke`, each
benchmark uses two examples; LongBench v2's raw prompt cap is 4,096 tokens and
generation is limited to eight tokens. Smoke accuracy is intentionally suppressed.

## 7. Run the complete GPU smoke matrix

This adds Llama, Qwen-14B and Qwen-72B, including multi-GPU execution where needed:

```bash
time bash scripts/run_phase_one.sh --gpus 5 --smoke \
  --out outputs/phase-one-smoke \
  2>&1 | tee outputs/logs/smoke-all.log
```

Check `outputs/phase-one-smoke/comparison.md` and the per-job logs. There should be
64 completed jobs before treating the whole matrix as ready. A job failure does
not prevent later jobs from running, but the launcher exits nonzero for failures.
An insufficient-memory job is recorded as unsupported; inspect the table as well
as the exit code. Repeat the exact command to resume after an interruption.

Passing this smoke run checks short-input execution only. It does not prove that
120K prompts, 16K outputs, all dataset examples, or the separate judge will work.
Native H200 execution has not yet been verified on the development machine.

## 8. Run the full benchmark

```bash
time bash scripts/run_phase_one.sh --gpus 5 \
  --out outputs/phase-one \
  2>&1 | tee outputs/logs/full-run.log
```

The launcher freezes revisions, datasets and chat token IDs, estimates memory,
and assigns disjoint GPU sets to jobs. Small models can run as independent jobs;
72B uses model layer sharding, not tensor parallelism. It detects the smallest
selected GPU's memory and reserves 10% when planning. There is no automatic
weight quantization or CPU/disk model offload. Memory estimates are not fit guarantees.

The full protocol uses a 120,000-token raw prompt cap for LongBench v2, up to 128
output tokens there, and up to 16,000 output tokens for LongGenBench. Do not add
`--smoke` to this command, and do not reuse a smoke output directory for a full run.

For a different GPU allocation, change both `CUDA_VISIBLE_DEVICES` and `--gpus N`.
Use `--gpu-memory-gib 130` only if deliberately choosing a lower planning limit;
keep that setting consistent on resume.

## 9. Monitor progress and resume

In another terminal, from the repository directory:

```bash
watch -n 5 nvidia-smi
```

Launcher output is in `outputs/logs/full-run.log`. Each worker also writes a log,
for example:

```bash
tail -f outputs/phase-one/results/Llama-3.1-8B-Instruct/longbenchv2/full/result.log
```

Checkpoints are atomic JSON files next to those logs. `plan.json` and the combined
table update as job results are collected; individual checkpoint rows provide
more immediate sample progress. Reading checkpoints is safe while workers run.
The separate `report` command takes an exclusive lock, so use it after the launcher
has stopped, not as a live monitor.

To resume generation, run the **same full-run command** again. Use `tee -a` to
append to the launcher log if desired. Finished examples are skipped and completed
jobs avoid reloading weights. An interrupted example restarts from its beginning;
mid-token KV state is not checkpointed. Only one launcher/judge can use an output
directory at a time, and its lock releases when the process terminates.

If source, environment or per-job GPU layout changes, resume may refuse the old
run. Use a new output directory for changed experiments rather than mixing results.

## 10. Judge LongGenBench

Once generation has finished and its GPUs are free:

```bash
time CUDA_VISIBLE_DEVICES=0 .envs/phase-judge/bin/python \
  -m experiments.phase_one_judge --out outputs/phase-one --gpus 1 \
  2>&1 | tee outputs/logs/longgen-judge.log
```

Here `0` is a physical GPU ID: change it if GPU 0 is not allocated to you.
The judge downloads/loads the pinned Qwen3-32B model on first use. This stage uses
FreeKV's evaluation prompts and settings, and commits each completed check. Repeat
the command to resume. It skips incomplete generation jobs and smoke outputs.
Generation completion alone does not validate the judge environment.

## 11. Generate and inspect the final tables

```bash
.envs/phase-baselines/bin/python -m experiments.phase_one report \
  --out outputs/phase-one
```

Results are in:

- `outputs/phase-one/comparison.md`: readable combined table.
- `outputs/phase-one/comparison.csv`: spreadsheet-ready table.
- `outputs/phase-one/comparison.json`: structured table.
- `outputs/phase-one/results/`: raw answers, token IDs, logs and judge checkpoints.
- `outputs/phase-one/revisions.json`, `manifests/`, `plan.json`: frozen inputs and allocation.

LongBench v2 accuracy and LongGenBench completion are percentages. LongGenBench
judged accuracy is reported on a 0–1 scale. Empty judged accuracy means the matching
judge stage is pending. Keep the complete output directory and environment locks
with your results; these generated artifacts are excluded from Git tracking.

## 12. How long will it take?

**Plan for days to weeks for the complete matrix, not a few hours.** There is no
measured H200 ETA yet. The current quality adapters run one sequence per worker;
72B occupies multiple GPUs, and CPU processing/model loading can leave GPUs idle.
Five GPUs do not imply a fivefold speedup for every method.

The configured LongGenBench workload is:

```text
4 models × 8 methods × 400 examples = 12,800 generations
12,800 × 16,000 max output tokens = 204,800,000 output tokens
```

The following are **arithmetic scenarios, not measured or promised H200 speeds**.
The rate is the total across all concurrently running workers on the five-GPU
machine; it is not a per-GPU rate. Early EOS/stop tokens can shorten generation.

| Assumed sustained aggregate output rate | If outputs average 8,000 tokens | If every output reaches 16,000 tokens |
|---|---:|---:|
| 100 tokens/second | 11.85 days | 23.70 days |
| 200 tokens/second | 5.93 days | 11.85 days |
| 500 tokens/second | 2.37 days | 4.74 days |
| 1,000 tokens/second | 1.19 days | 2.37 days |

This table covers **LongGenBench decoding only**. Add installation, downloads,
all prompt prefills, model reloads, codec calibration/compression, scheduling gaps,
LongBench v2 and judging. The real run can be slower or exceed these time ranges.
The [LongBench v2 dataset card](https://huggingface.co/datasets/zai-org/LongBench-v2)
currently lists 503 examples: another 16,096 evaluations across 32 model/method
pairs, with potentially very long input contexts. The runner uses the actual
frozen split size; it does not hard-code 503.

| Stage | What can be estimated now |
|---|---|
| Setup/downloads | Network and compilation dependent. Transferring 270 GB alone takes about 6 hours at a sustained 100 Mbit/s, or 36 minutes at 1 Gbit/s, before overhead and build time. |
| Regression tests | Recorded development CPU durations: 48.92 seconds + 13.90 seconds for the two suites. Server durations may differ. |
| GPU smoke tests | Use the `time` output. Short generation is cheap, but first downloads, calibration and repeated 72B loading can dominate. No H200 measurement is available. |
| Full generation | Days-to-weeks planning scope; use the conditional table and actual per-job timings. There is no validated upper runtime bound. |
| LongGenBench judging | Additional time; depends on how many blocks/checks each output contains and judge throughput. It is not included in the decoding table. |

### Refine the estimate while the real run progresses

Do not scale up the eight-token smoke test to predict a 16,000-token run. Let at
least 5–10 **full-length** examples finish for each model/method/benchmark, then
inspect their recorded durations. This command is read-only and can run in a
second terminal while the launcher is active:

```bash
.envs/phase-baselines/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('outputs/phase-one')
for path in sorted((root / 'results').glob('*/*/*/result.json')):
    result = json.loads(path.read_text())
    rows = result.get('rows', [])
    expected = len(result.get('expected', []))
    label = path.parent.relative_to(root / 'results')
    print(f'{label}: {len(rows)}/{expected}, status={result.get("status")}')
    if result.get('smoke') or len(rows) < 5:
        print('  Insufficient full-run samples for an estimate.')
        continue
    # Exclude the first sample, which can include one-time codec calibration.
    recent = rows[1:][-10:]
    mean_seconds = sum(r['elapsed_seconds'] for r in recent) / len(recent)
    mean_tokens = sum(r['output_tokens'] for r in recent) / len(recent)
    remaining_hours = max(0, expected-len(rows)) * mean_seconds / 3600
    print(f'  Recent mean: {mean_seconds:.1f} s/sample, {mean_tokens:.0f} output tokens')
    print(f'  This job alone: about {remaining_hours:.1f} h remaining at that pace')
PY
```

These are per-job extrapolations, not the total suite ETA. Prompt lengths vary;
jobs not yet started are missing; sample time excludes model loading; scheduler
waves and GPU reservations affect wall time. Use representative short/long cases
and observed whole-stage wall time before committing to a fixed completion date.

## Important interpretation limits

The full implementation/protocol details are in [PHASE_ONE_RUNBOOK.md](PHASE_ONE_RUNBOOK.md).
In particular, ours currently compresses the prompt archive but retains generated
KV, and H2O is KVCache-Factory's prefill variant. The Qwen runs follow FreeKV's
explicit unscaled long-context protocol. The outputs are quality results; this
pipeline does not establish the latency/memory claims in the illustrative figures.
