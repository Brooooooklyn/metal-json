#!/usr/bin/env bash
# CI convenience wrapper: fetch canonical benchmark datasets into data/bench/.
set -euo pipefail
cd "$(dirname "$0")/.."
exec cargo run -p xtask --quiet -- fetch-data "$@"
