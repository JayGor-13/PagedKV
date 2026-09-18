#!/usr/bin/env bash
# Run on the Linux H200 host, with a CUDA 12.4 toolkit and Python 3.10/3.11.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3.11}"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${MAX_JOBS:-8}"
"$PYTHON" -m venv .envs/phase-baselines
.envs/phase-baselines/bin/python -m pip install --upgrade pip wheel setuptools packaging ninja
.envs/phase-baselines/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.envs/phase-baselines/bin/python -m pip install transformers==4.45.2 accelerate==0.34.2 datasets==2.21.0 huggingface-hub==0.25.2 numpy==1.26.4 scipy sentencepiece einops tqdm pandas safetensors pytest
.envs/phase-baselines/bin/python -m pip install flash-attn==2.6.3 --no-build-isolation
.envs/phase-baselines/bin/python -m pip install flashinfer-python==0.2.4
.envs/phase-baselines/bin/python -m scripts.fetch_baselines --methods freekv factory rocketkv
"$PYTHON" -m venv .envs/phase-ours
.envs/phase-ours/bin/python -m pip install --upgrade pip
.envs/phase-ours/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.envs/phase-ours/bin/python -m pip install -r requirements-model.txt sentencepiece
"$PYTHON" -m venv .envs/phase-judge
.envs/phase-judge/bin/python -m pip install --upgrade pip
.envs/phase-judge/bin/python -m pip install vllm==0.10.2 transformers==4.55.2 huggingface-hub==0.34.4
mkdir -p outputs/environment-locks
for env in phase-baselines phase-ours phase-judge; do
  ".envs/$env/bin/python" -m pip freeze > "outputs/environment-locks/$env.txt"
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > outputs/environment-locks/gpus.csv
echo "Environments prepared. Run the smoke commands in PHASE_ONE_RUNBOOK.md before the full suite."
