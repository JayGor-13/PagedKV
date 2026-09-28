#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.envs/phase-ours/bin/python -m pytest tests -q -p no:cacheprovider
.envs/phase-baselines/bin/python -m pytest tests/test_phase_one_backend_math.py -q -p no:cacheprovider
.envs/phase-baselines/bin/python -m scripts.fetch_baselines --methods freekv factory rocketkv --verify-only
.envs/phase-ours/bin/python -m scripts.smoke --output outputs/cpu-codec-smoke.json
.envs/phase-ours/bin/python -m scripts.smoke_model_integration
revision="$(git rev-parse --short HEAD)"
.envs/phase-ours/bin/python -m experiments.phase_one_cpu_rehearsal \
  --out "outputs/cpu-phase-one-rehearsal-$revision"
echo "CPU regression, model-wiring, and 64-job orchestration tests passed. Native H200 validation is the separate --smoke run."
