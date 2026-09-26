# H200 commands: clone to final results

Run this on the Linux server. It assumes five allocated H200 GPUs with IDs 0–4.
Change the IDs once at the top if the allocation differs. Do not change source,
environments, GPU count, or output directory while resuming a run.

## 1. Clone and select the GPUs

```bash
git clone https://github.com/JayGor-13/PagedKV.git
cd PagedKV
git rev-parse HEAD

export CUDA_VISIBLE_DEVICES=0,1,2,3,4
mkdir -p outputs/logs
set -o pipefail
nvidia-smi
```

Use `tmux new -s pagedkv` before the long commands when running through SSH.

## 2. Install, authenticate, and verify the machine

```bash
time bash scripts/setup_phase_one.sh 2>&1 | tee outputs/logs/setup.log

# Optional explicit interpreter when python3.11 is unavailable:
# PYTHON=/usr/bin/python3.12 bash scripts/setup_phase_one.sh

.envs/phase-baselines/bin/python -c "from huggingface_hub import login; login()"
.envs/phase-baselines/bin/python -c "from huggingface_hub import whoami; print(whoami())"

bash scripts/preflight_h200.sh --gpus 5 --online \
  --out outputs/h200-preflight.json \
  2>&1 | tee outputs/logs/preflight.log

time bash scripts/test_phase_one.sh 2>&1 | tee outputs/logs/tests.log
```

Stop if setup, preflight, or tests return a nonzero exit status. The Hugging Face
account must have access to `meta-llama/Llama-3.1-8B-Instruct`.

## 3. Run the one-GPU 7B smoke test

```bash
time bash scripts/run_phase_one.sh --gpus 1 --smoke \
  --models Qwen/Qwen2.5-7B-Instruct \
  --out outputs/phase-one-smoke-7b \
  2>&1 | tee outputs/logs/smoke-7b.log
```

## 4. Freeze and run the complete 64-job smoke matrix

```bash
bash scripts/run_phase_one.sh --gpus 5 --smoke --plan-only \
  --require-complete-plan --out outputs/phase-one-smoke

time bash scripts/run_phase_one.sh --gpus 5 --smoke \
  --require-complete-plan --out outputs/phase-one-smoke \
  2>&1 | tee outputs/logs/smoke-all.log
```

Open `outputs/phase-one-smoke/comparison.md`. All 64 rows must be completed before
starting production. Rerun the identical second command to resume a stopped smoke.

## 5. Freeze and run the complete production matrix

```bash
bash scripts/run_phase_one.sh --gpus 5 --plan-only \
  --require-complete-plan --out outputs/phase-one

time bash scripts/run_phase_one.sh --gpus 5 \
  --require-complete-plan --out outputs/phase-one \
  2>&1 | tee outputs/logs/full-run.log
```

This runs Llama-3.1-8B-Instruct and Qwen2.5-7B/14B/72B-Instruct on LongBench v2
and LongGenBench with full cache, H2O, SnapKV, Quest, ArkVale, RocketKV, FreeKV,
and ours. To resume after an interruption, repeat the exact production command and
append the launcher log:

```bash
time bash scripts/run_phase_one.sh --gpus 5 \
  --require-complete-plan --out outputs/phase-one \
  2>&1 | tee -a outputs/logs/full-run.log
```

Monitor from another terminal with `watch -n 5 nvidia-smi`. Per-job logs are below
`outputs/phase-one/results/`.

## 6. Judge LongGenBench and write final tables

Run this only after generation has stopped and the GPUs are free:

```bash
CUDA_VISIBLE_DEVICES=0 .envs/phase-judge/bin/python \
  -m experiments.phase_one_judge --out outputs/phase-one --gpus 1 \
  2>&1 | tee outputs/logs/longgen-judge.log

.envs/phase-baselines/bin/python -m experiments.phase_one report \
  --out outputs/phase-one
```

Repeat the judge command with `tee -a` to resume it. Final combined results are
`outputs/phase-one/comparison.md`, `comparison.csv`, and `comparison.json`.
