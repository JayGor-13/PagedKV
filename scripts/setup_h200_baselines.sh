#!/usr/bin/env bash
# Run on Linux/H200 with Python 3.10 and a CUDA 12.x development toolkit.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PYTHON="${PYTHON:-python3.10}"
ENVROOT="${ENVROOT:-$ROOT/.envs}"
METHOD="${1:-kivi}"
if [[ "$METHOD" != kivi && "$METHOD" != arkvale ]]; then
  echo 'Usage: bash scripts/setup_h200_baselines.sh kivi|arkvale' >&2
  exit 2
fi
"$PYTHON" -m scripts.fetch_baselines --methods "$METHOD"
"$PYTHON" -m venv "$ENVROOT/$METHOD"
PY="$ENVROOT/$METHOD/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
"$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
"$PY" -m pip install 'numpy<2' ninja 'cmake>=3.26.4' packaging sentencepiece protobuf accelerate scipy scikit-learn einops
if [[ "$METHOD" == kivi ]]; then
  "$PY" -m pip install transformers==4.43.1
else
  "$PY" -m pip install transformers==4.40.0
fi
"$PY" -m pip install flash-attn==2.6.3 --no-build-isolation
"$PY" -m pip install -r requirements-evaluation.txt
export TORCH_CUDA_ARCH_LIST=9.0
if [[ "$METHOD" == kivi ]]; then
  # Dependencies installed above deliberately; upstream otherwise re-resolves torch.
  "$PY" -m pip install --no-deps -e external/KIVI
  (cd external/KIVI/quant && "$PY" -m pip install --no-build-isolation --no-deps -e .)
  "$PY" -c 'import kivi_gemv; from models.llama_kivi import LlamaForCausalLM_KIVI'
else
  (cd external/ArkVale/source && "$PY" -m pip install --no-build-isolation --no-deps -e .)
  "$PY" -c 'import arkvale_cpp; from arkvale import adapter'
fi
"$PY" -m pip freeze > "$ENVROOT/$METHOD/freeze.txt"
echo "Environment ready: $PY"
echo 'This verifies installation/imports only. Run a short GPU correctness job before long experiments.'
