#!/usr/bin/env bash
# Checks everything the `gpu` CI job (and the GPU tests anywhere) needs and prints the versions it
# found. Exits non-zero listing ALL problems, so a new runner is set up in one go.
#   scripts/ci_gpu_preflight.sh
set -uo pipefail

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
LLVM_DIR_PREFIX=${QUANTFORGE_LLVM_PREFIX:-/usr/lib/llvm-18}
problems=0
ok()   { printf '  ok    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; problems=$((problems + 1)); }
need() { # need <command> <hint>
  if command -v "$1" > /dev/null 2>&1; then ok "$1: $(command -v "$1")"; else bad "$1 not found ($2)"; fi
}

echo "== build toolchain"
need cmake "apt install cmake (>= 3.21)"
need ninja "apt install ninja-build"
need g++ "apt install g++ (also builds the torch extension)"
[ -d "$LLVM_DIR_PREFIX/lib/cmake/mlir" ] && ok "MLIR: $LLVM_DIR_PREFIX/lib/cmake/mlir" \
  || bad "MLIR 18 not found under $LLVM_DIR_PREFIX (apt install llvm-18-dev libmlir-18-dev mlir-18-tools)"
for t in mlir-translate llc; do
  [ -x "$LLVM_DIR_PREFIX/bin/$t" ] && ok "$t: $LLVM_DIR_PREFIX/bin/$t" || bad "$t missing in $LLVM_DIR_PREFIX/bin"
done

echo "== CUDA"
need nvidia-smi "NVIDIA driver"
if command -v nvidia-smi > /dev/null 2>&1; then
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -1)
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | head -1)
  if [ -z "$cc" ]; then bad "nvidia-smi sees no GPU"
  elif awk -v c="$cc" 'BEGIN { exit !(c >= 8.0) }'; then ok "GPU: $name (sm_${cc/./})"
  else bad "GPU $name has compute capability $cc; the kernels need >= 8.0 (cp.async, mma.sync m16n8k16)"; fi
fi
[ -x "$CUDA_HOME/bin/ptxas" ] && ok "ptxas: $("$CUDA_HOME/bin/ptxas" --version | tail -1)" \
  || bad "ptxas not found under CUDA_HOME=$CUDA_HOME (set CUDA_HOME)"
[ -f "$CUDA_HOME/include/cuda.h" ] && ok "CUDA headers: $CUDA_HOME/include" || bad "cuda.h missing under $CUDA_HOME/include"
SAN=${COMPUTE_SANITIZER:-$CUDA_HOME/bin/compute-sanitizer}
[ -x "$SAN" ] || SAN=$(command -v compute-sanitizer || true)
[ -n "$SAN" ] && [ -x "$SAN" ] && ok "compute-sanitizer: $SAN" || bad "compute-sanitizer not found (scripts/sanitize.sh)"

echo "== python (python3 = $(command -v python3 || echo none))"
if command -v python3 > /dev/null 2>&1; then
  python3 - << 'PY' || problems=$((problems + 1))
import sys
ok = lambda m: print(f"  ok    {m}")
bad = lambda m: (print(f"  FAIL  {m}"), sys.exit(1))
try:
    import torch
except Exception as e:  # noqa: BLE001
    bad(f"import torch failed: {e}")
if not torch.cuda.is_available():
    bad("torch.cuda.is_available() is False")
ok(f"torch {torch.__version__} (CUDA {torch.version.cuda}), device: {torch.cuda.get_device_name(0)}")
try:
    import pytest
    ok(f"pytest {pytest.__version__}")
except Exception as e:  # noqa: BLE001
    bad(f"pytest missing: {e}")
for opt, why in (("vllm", "Marlin comparison / real-checkpoint oracle (those tests skip without it)"),
                 ("transformers", "benchmarks/qwen_*.py only")):
    try:
        __import__(opt)
        ok(f"{opt} importable")
    except Exception:  # noqa: BLE001
        print(f"  note  {opt} not installed: {why}")
PY
fi

echo "== resources"
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2> /dev/null || echo 0)
if [ "${avail_kb:-0}" -ge 8000000 ]; then ok "MemAvailable $((avail_kb / 1024)) MiB"
else bad "MemAvailable $((avail_kb / 1024)) MiB < 8 GiB (the build uses ~0.6 GiB per job: cap with -j)"; fi
free_gb=$(df -Pk . | awk 'NR==2 {print int($4 / 1048576)}')
[ "$free_gb" -ge 5 ] && ok "free disk: ${free_gb} GiB" || bad "free disk ${free_gb} GiB < 5 GiB"

echo
if [ "$problems" -eq 0 ]; then echo "preflight: all good"; else echo "preflight: $problems problem(s)"; fi
exit $((problems > 0))
