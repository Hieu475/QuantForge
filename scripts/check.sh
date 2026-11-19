#!/usr/bin/env bash
# Configure, build and test QuantForge.   scripts/check.sh [--gpu] [--sanitize]
#
#   (default)   build + lit tests (CPU only)
#   --gpu       also run the GPU end-to-end tests (needs NVIDIA GPU + torch)
#   --sanitize  also run a smoke test under compute-sanitizer (memcheck+racecheck)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=0; SAN=0
for a in "$@"; do
  case "$a" in
    --gpu) GPU=1 ;;
    --sanitize) SAN=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

PRESET=${QUANTFORGE_PRESET:-default}
BUILD_DIR=${QUANTFORGE_BUILD_DIR:-build}
# the "default" preset builds into ./build; "ci" into ./build-ci
[ "$PRESET" = ci ] && BUILD_DIR=${QUANTFORGE_BUILD_DIR:-build-ci}

cmake --preset "$PRESET"
cmake --build --preset "$PRESET"
cmake --build "$BUILD_DIR" --target check-quantforge

if [ "$GPU" = 1 ]; then
  QUANTFORGE_BUILD_DIR="$PWD/$BUILD_DIR" python3 -m pytest -v
fi
if [ "$SAN" = 1 ]; then
  scripts/sanitize.sh
fi
