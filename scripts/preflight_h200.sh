#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .envs/phase-baselines/bin/python -m scripts.h200_preflight "$@"
