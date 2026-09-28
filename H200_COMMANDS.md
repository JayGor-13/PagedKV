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

Your server currently reports **CUDA toolkit 13.1** from `nvcc`. The pinned
baseline compiles FlashAttention against PyTorch `cu124`, so first select a CUDA
12.x toolkit; 12.4 is the reproducible target. Keep the current NVIDIA driver.

```bash
# Use the command supported by your server (ask the administrator if neither exists):
module avail cuda
module load cuda/12.4

# Alternative when CUDA 12.4 is installed under /usr/local:
# export CUDA_HOME=/usr/local/cuda-12.4
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

hash -r
nvcc --version                 # must now say release 12.x; target 12.4
```

Before installing anything, the dependency-free CPU rehearsal can verify the
64-cell plan, scheduling, checkpoint/resume, and report path:

```bash
python3 -m experiments.phase_one_cpu_rehearsal \
  --out "outputs/cpu-rehearsal-$(git rev-parse --short HEAD)"
cat "outputs/cpu-rehearsal-$(git rev-parse --short HEAD)/rehearsal-summary.json"
```

This does not load models or validate CUDA/native kernels, accuracy, or latency.

Use `tmux new -s pagedkv` before the long commands when running through SSH.

## 2. Install, authenticate, and verify the machine

```bash
time bash scripts/setup_phase_one.sh 2>&1 | tee outputs/logs/setup.log

# Optional explicit interpreter when python3.11 is unavailable:
# PYTHON=/usr/bin/python3.12 bash scripts/setup_phase_one.sh

.envs/phase-baselines/bin/python -c "from huggingface_hub import login; login()"
.envs/phase-baselines/bin/python -c "from huggingface_hub import whoami; print(whoami())"

# Alternatively, set a read token for this shell without placing it in history:
# read -r -s -p 'Hugging Face read token: ' HF_TOKEN; echo; export HF_TOKEN

# Real pretrained model + real dataset CPU check (one shortened example each):
.envs/phase-baselines/bin/python -m scripts.real_hf_cpu_smoke \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --check-models meta-llama/Llama-3.1-8B-Instruct \
    Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct \
    Qwen/Qwen2.5-72B-Instruct \
  --out outputs/real-hf-cpu-smoke.json

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
