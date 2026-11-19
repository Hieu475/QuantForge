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
