#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.envs/phase-ours/bin/python -m pytest tests -q -p no:cacheprovider
.envs/phase-baselines/bin/python -m pytest tests/test_phase_one_backend_math.py -q -p no:cacheprovider
.envs/phase-baselines/bin/python -m scripts.fetch_baselines --methods freekv factory rocketkv --verify-only
echo "CPU/integration tests passed. Native H200 validation is the separate --smoke run."
