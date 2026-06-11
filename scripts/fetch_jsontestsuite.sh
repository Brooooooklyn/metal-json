#!/usr/bin/env bash
# Fetch the nst/JSONTestSuite conformance corpus into data/JSONTestSuite.
#
# Idempotent: if the corpus is already present, this is a no-op. data/ is
# gitignored; tests/jsontestsuite.rs and tests/differential.rs auto-skip
# (loudly) when the corpus is missing, so running this script is optional
# locally but required for full coverage (CI runs it before the test step).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/JSONTestSuite"

if [ -d "$DEST/test_parsing" ]; then
    echo "JSONTestSuite already present at $DEST — nothing to do."
    exit 0
fi

mkdir -p "$ROOT/data"
git clone --depth 1 https://github.com/nst/JSONTestSuite "$DEST"
echo "Fetched JSONTestSuite into $DEST."
