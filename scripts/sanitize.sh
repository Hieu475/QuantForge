#!/usr/bin/env bash
# Run a small GEMM under compute-sanitizer (memcheck: out-of-bounds / misaligned
# accesses; racecheck: shared-memory hazards, i.e. missing barriers) for the main
# pipeline variants. Needs a GPU. Exit code != 0 on any finding or mismatch.
set -euo pipefail
cd "$(dirname "$0")/.."
export QUANTFORGE_BUILD_DIR=${QUANTFORGE_BUILD_DIR:-$PWD/build}
SAN=${COMPUTE_SANITIZER:-/usr/local/cuda/bin/compute-sanitizer}
[ -x "$SAN" ] || SAN=$(command -v compute-sanitizer)

fail=0
# <pipeline options>|M K N G bits   (G 0 = per-tensor). The group-wise and decode
# rows exercise the 16-byte vector loads of scale / zero point / packed weights:
# memcheck reports any misaligned or out-of-bounds access.
CASES=(
  "|256 128 256 0 4"
  "smem-stages=1|256 128 256 0 4"
  "tensor-core=false|256 128 256 0 4"
  "async-copy=false vector-copy=false|256 128 256 0 4"
  "|256 128 256 64 4"
  "vectorize-loads=false|256 128 256 64 4"
  "decode=true|16 256 128 0 4"
  "decode=true|16 512 256 128 4"
  "decode=true smem-stages=2|16 256 128 64 4"
  "|128 256 128 64 8"
  "decode=true|16 256 128 0 8"
  # FP8 E4M3 weights (kind 9)
  "|128 256 128 64 9"
  "decode=true|16 512 256 128 9"
  "decode=true block-tile-n=128 warp-tile-n=16 block-tile-k=64 smem-stages=1 prefetch-distance=1|16 512 256 128 9"
  # multi-tile weight-load look-ahead (rotating registers; reads past the last tile are clamped,
  # also when the K loop is shorter than the distance), small K tile, fused activations
  "decode=true block-tile-n=128 warp-tile-n=16 block-tile-k=64 smem-stages=1 prefetch-distance=2|16 512 256 128 4"
  "decode=true block-tile-k=128 smem-stages=1 prefetch-distance=3|16 256 128 128 4"
  "decode=true smem-stages=2 sink-dequant-stores=true prefetch-distance=1|16 512 128 64 4"
  "decode=true epilogue=silu|16 256 128 64 4"
  "epilogue=gelu|128 128 128 64 4"
  "block-tile-m=64 block-tile-n=128 warp-tile-m=32 warp-tile-n=64 block-tile-k=64 smem-stages=1|128 256 256 128 4"
  # C = A @ W (no initial read of C), alone and with the look-ahead registers; fused gate / up
  "decode=true accumulate=false prefetch-distance=2 smem-stages=1|16 512 256 128 4"
  "accumulate=false|128 128 128 64 4"
  "decode=true epilogue=swiglu|16 256 256 64 4"
  "decode=true epilogue=swiglu prefetch-distance=1 smem-stages=1 block-tile-n=128 warp-tile-n=16|16 512 256 128 4"
  "epilogue=swiglu|128 128 256 64 4"
)
for c in "${CASES[@]}"; do
  opts=${c%%|*}; shape=${c#*|}
  for tool in memcheck racecheck; do
    echo "=== $tool  opts='$opts' shape='$shape'"
    # shellcheck disable=SC2086
    if ! "$SAN" --tool "$tool" --error-exitcode 1 \
         python3 python/quantforge/smoke.py "$opts" $shape; then
      echo "FAILED: $tool opts='$opts' shape='$shape'"; fail=1
    fi
  done
done
exit $fail
