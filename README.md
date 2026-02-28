# QuantForge

<div align="center">

**MLIR-based Compiler for INT4 Quantized Neural Network Inference**

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![MLIR](https://img.shields.io/badge/MLIR-LLVM-orange.svg)](https://mlir.llvm.org/)
[![CUDA](https://img.shields.io/badge/CUDA-PTX-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## 🎯 Overview

QuantForge is a specialized compiler built on MLIR/LLVM infrastructure, designed to **maximize effective memory bandwidth** of GEMV operations by fusing INT4 unpacking directly into GPU register files.

### The Problem

During LLM decode phase, the bottleneck is **HBM read speed**, not Tensor Core compute. Traditional frameworks perform:

```
HBM → Unpack to FP16 → Write back to HBM → Matrix multiply
```

This creates redundant memory transactions that destroy quantization benefits.

### Our Solution

QuantForge generates PTX assembly with **on-the-fly unpacking**:

```
HBM → Shared Memory → Registers → bit-shift + mask → mma.sync/fma
```

Benefits:
- **2-4x** more parameters loaded per clock cycle vs FP16
- Zero framework overhead
- Near-theoretical memory bandwidth utilization

---

## 🏗️ Architecture

```
Input Graph (TOSA/Linalg)
        │
        ▼
┌─────────────────┐
│  QuantForge     │  qf.unpack, qf.dequant
│  Dialect        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tiling &       │  128×128 tiles
│  Vectorization  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Dialect │  arith.shrui, arith.andi
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GPU/NVVM       │  gpu.launch_func
│  Dialect        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PTX/CUBIN      │  Native GPU assembly
└─────────────────┘
```

---

## 📦 QuantForge Dialect

### `qf.unpack` - Unpack INT4 Weights

```mlir
%unpacked = qf.unpack %packed : tensor<4096x2048xi8> -> tensor<4096x4096xi8>
```

Extracts two 4-bit values from each INT8 byte using bitwise operations.

### `qf.dequant` - Dequantize to FP16

```mlir
%fp = qf.dequant %int, %scale, %zp : tensor<4096x4096xi8>, tensor<f16>, tensor<i8> -> tensor<4096x4096xf16>
```

Applies: `output = (input - zero_point) * scale`

---

## 🚀 Quick Start

### Prerequisites

- LLVM/MLIR 17+
- CMake 3.20+
- C++17 compiler
- CUDA Toolkit (for GPU codegen)
- Python 3.8+ with PyTorch

### Build

```bash
mkdir build && cd build
cmake .. -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
         -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
cmake --build . -j$(nproc)
```

### Run

```bash
# Run optimizer
./build/tools/quantforge-opt/quantforge-opt test/Dialect/QuantForge/basic.mlir

# Run baseline benchmark
python python/quantforge/baseline.py
```

---

## 📁 Project Structure

```
QuantForge/
├── include/QuantForge/
│   ├── Dialect/QuantForge/    # Dialect & Ops definitions (.td, .h)
│   └── Transforms/            # Pass declarations
├── lib/
│   ├── Dialect/QuantForge/    # Dialect implementation (.cpp)
│   └── Transforms/            # Pass implementations
├── tools/quantforge-opt/      # Main optimizer driver
├── python/quantforge/         # Python baseline & bindings
└── test/                      # MLIR test cases
```

---

## 🎯 Performance Targets

| Metric | Target |
|--------|--------|
| Correctness | cosine similarity > 0.99 vs FP16 |
| Self-containment | Zero cuBLAS/cuDNN calls |
| Latency | 1.5-2x faster than PyTorch naive |
| Bandwidth | > 75% theoretical peak |
| Register Spilling | 0 bytes |

---

## 📚 W4A16 Quantization

```python
# Pack: 2 INT4 → 1 INT8
packed = low | (high << 4)

# Unpack: 1 INT8 → 2 INT4
low  = packed & 0x0F
high = (packed >> 4) & 0x0F

# Dequantize: INT4 → FP16
fp16 = (int4 - zero_point) * scale
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Compiler | MLIR/LLVM |
| GPU | CUDA/PTX |
| Language | C++17 |
| Testing | Python/PyTorch |
| Build | CMake |

---

## 📖 References

- [MLIR Documentation](https://mlir.llvm.org/)
- [CUDA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [LLVM TableGen](https://llvm.org/docs/TableGen/)

---

## 📄 License

MIT License