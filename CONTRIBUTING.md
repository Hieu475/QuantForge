# Contributing

## Dev loop

```bash
cmake --preset default && cmake --build --preset default
cmake --build build --target check-quantforge        # fast, CPU only
QUANTFORGE_BUILD_DIR=$PWD/build python -m pytest -v   # GPU end-to-end
```

`scripts/check.sh --gpu --sanitize` runs all of it.

## Rules of thumb

* **Every pass gets a lit test** (`test/Transforms/*.mlir`, FileCheck). Test the
  *behaviour* (what is rewritten, what is left alone), not just that it runs.
* **Anything that changes generated code must also pass the E2E tests** on a GPU:
  FileCheck cannot see a wrong result. `test/E2E/test_w4a16_e2e.py` runs the
  pipeline under many option combinations and compares with PyTorch; add the new
  option/config to its `CONFIGS` list.
* A change that can affect synchronisation or shared-memory layout also needs
  `scripts/sanitize.sh` to stay clean (racecheck catches missing barriers).
* **Performance changes need numbers**: `python python/quantforge/benchmark.py 2048 "" "<new option>"`
  (best-of, round-robin between configurations — wall-clock timing on laptop
  GPUs drifts with temperature). Profile with `ncu` before guessing.
  Update `benchmarks/baseline_*.json` only when a regression is intentional or an
  improvement is permanent, and say so in the commit message.
* LLVM-version workarounds belong next to a comment that says which LLVM bug they
  avoid and how to detect that it is fixed.
* Formatting: LLVM style (`.clang-format`); format only what you touch
  (`git clang-format <base>`). Python: black-ish, 100 columns.

## Supported toolchain

LLVM/MLIR 18.1.x only. Moving to a newer MLIR is a deliberate port (the
`scf.forall`→`gpu.launch`, bufferization and nvgpu APIs all moved), not a flag.

## Commit messages

`area: what changed` + a body explaining *why* and how it was verified, e.g.
`smem: double-buffer the K loop (+5% at 2048³)`.

By contributing you agree that your contribution is licensed under the project's
license (Apache-2.0 with LLVM Exceptions, see `LICENSE`).

## Setting up the GPU CI runner

The `gpu` job of `.github/workflows/ci.yml` runs on a self-hosted runner with the labels
`self-hosted, linux, gpu`. A machine is ready when `scripts/ci_gpu_preflight.sh` prints
`preflight: all good`; it checks, and names the missing package for:

* cmake >= 3.21, ninja, g++ (the build and the PyTorch C++ extension), `llvm-18-dev libmlir-18-dev mlir-18-tools`
* an NVIDIA GPU with compute capability >= 8.0, the driver, `CUDA_HOME` (default `/usr/local/cuda`) with
  `ptxas`, the CUDA headers and `compute-sanitizer`
* the `python3` on the runner's PATH with `torch` (CUDA build) and `pytest`; `vllm` and `transformers`
  are optional (the Marlin comparison, the real-checkpoint test and the Qwen benchmarks skip without them)
* >= 8 GiB of available RAM (`QUANTFORGE_BUILD_JOBS` caps the build parallelism, default 6) and 5 GiB of disk

The job runs: preflight, build (preset `default`), `pytest`, `scripts/sanitize.sh` and, nightly, the three
regression comparisons against `benchmarks/baseline_*.json` (15 % tolerance). The baselines were recorded
on an RTX 4050 Laptop GPU: on another GPU `benchmark.py --compare` warns and the comparison is meaningless
until the baselines are regenerated there (`benchmark.py ... --json benchmarks/baseline_*.json`).
First run populates `~/.cache/quantforge/kernels` and `~/.cache/torch_extensions` (a few minutes, then ~5 s).
