#!/usr/bin/env bash
# Dedicated small-model FP16/SDPA environments; no FlashAttention/FlashInfer builds.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), "Use Python 3.11 or 3.12"'
"$PYTHON" -m pip install virtualenv
for name in t4-baselines t4-ours; do
  "$PYTHON" -m virtualenv ".envs/$name"
  ".envs/$name/bin/python" -m pip install --upgrade pip wrapt
  ".envs/$name/bin/python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
done
.envs/t4-baselines/bin/python -m pip install transformers==4.45.2 accelerate==0.34.2 datasets==2.21.0 huggingface-hub==0.25.2 numpy==1.26.4 scipy sentencepiece einops pytest
.envs/t4-ours/bin/python -m pip install -r requirements-model.txt sentencepiece
.envs/t4-baselines/bin/python -m scripts.fetch_baselines --methods freekv factory rocketkv
mkdir -p outputs/t4-environment
for name in t4-baselines t4-ours; do
  ".envs/$name/bin/python" -m pip freeze > "outputs/t4-environment/$name.txt"
  ".envs/$name/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check accelerator and driver"; print(torch.cuda.get_device_name(0)); x=torch.ones(4, device="cuda", dtype=torch.float16); print(x @ x)'
done
nvidia-smi > outputs/t4-environment/nvidia-smi.txt
echo 'Ready for the Kaggle T4 smoke test. See KAGGLE_T4.md.'
