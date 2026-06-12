#!/usr/bin/env bash
# Re-fetch the vendored simdjson amalgamation reproducibly.
#
# Usage: scripts/update_simdjson_amalgamation.sh [tag]
#
# The tag is pinned below; pass an explicit tag (e.g. v4.7.0) to upgrade.
# Records the tag + sha256 of both files in bench/cpp/vendor/VERSION.
set -euo pipefail

PINNED_TAG="v4.6.4"
TAG="${1:-$PINNED_TAG}"

VENDOR_DIR="$(cd "$(dirname "$0")/.." && pwd)/bench/cpp/vendor"
mkdir -p "$VENDOR_DIR"
cd "$VENDOR_DIR"

for f in simdjson.h simdjson.cpp; do
  echo "fetching $f @ $TAG ..."
  curl -fSL --retry 3 -o "$f.part" \
    "https://github.com/simdjson/simdjson/releases/download/${TAG}/${f}"
  mv "$f.part" "$f"
done

{
  echo "simdjson amalgamation (https://github.com/simdjson/simdjson/releases)"
  echo "tag: ${TAG}"
  echo "sha256:"
  shasum -a 256 simdjson.h simdjson.cpp
  echo
  echo "Refresh with: scripts/update_simdjson_amalgamation.sh [tag]"
} > VERSION

echo "updated $VENDOR_DIR to ${TAG}"
cat VERSION
