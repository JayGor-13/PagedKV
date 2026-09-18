#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Examples: bash scripts/run_phase_one.sh --gpus 5
#           bash scripts/run_phase_one.sh --gpus 5 --smoke --out outputs/phase-one-smoke
exec .envs/phase-baselines/bin/python -m experiments.phase_one run "$@"
