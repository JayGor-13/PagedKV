#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.envs/t4-ours/bin/python -m pytest tests -q -p no:cacheprovider
.envs/t4-baselines/bin/python -m pytest tests/test_phase_one_backend_math.py tests/test_kaggle_t4.py -q -p no:cacheprovider
.envs/t4-baselines/bin/python -m scripts.fetch_baselines --methods freekv factory rocketkv --verify-only
