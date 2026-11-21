# syntax=docker/dockerfile:1
# QuantForge with everything built: the quantforge-opt compiler (LLVM/MLIR 18), a CUDA build of PyTorch, the Python package
# and, optionally, vLLM with the `quantization="quantforge"` plugin.
#
#   docker build -f docker/Dockerfile -t quantforge .                       # compiler + torch + package
#   docker build -f docker/Dockerfile --build-arg WITH_VLLM=1 -t quantforge:vllm .
#   docker run --rm --gpus all quantforge                                   # runs quantforge-doctor
#   docker run --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface quantforge:vllm python my_script.py
#
# Needs the NVIDIA container toolkit on the host; the image carries the CUDA toolkit (nvcc / ptxas are used at run time).
ARG CUDA_VERSION=12.8.1

# ---- stage 1: build the compiler ------------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build python3 python3-venv llvm-18-dev libmlir-18-dev mlir-18-tools \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# the CI preset: Release, no GPU tests registered; -j capped because the MLIR translation units are memory hungry
RUN cmake --preset ci && cmake --build build-ci --target quantforge-opt --parallel 6

# ---- stage 2: run time ----------------------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04
ARG WITH_VLLM=0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.11.0
# llvm-18 / mlir-18-tools: quantforge-opt links libLLVM.so.18 and the harness runs llc and mlir-translate;
# g++ + ninja: the torch C++ op is JIT-built on first import
RUN apt-get update && apt-get install -y --no-install-recommends \
        llvm-18 mlir-18-tools g++ ninja-build python3 python3-venv python3-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=compiler /src/build-ci/tools/quantforge-opt/quantforge-opt /usr/local/bin/quantforge-opt
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH CUDA_HOME=/usr/local/cuda
# vLLM pins its own torch: install it first so that pip picks a consistent torch, otherwise use the CUDA 12.8 build of torch
RUN if [ "$WITH_VLLM" = "1" ]; then pip install --no-cache-dir vllm; \
    else pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url ${TORCH_INDEX}; fi
COPY python /opt/quantforge/python
RUN pip install --no-cache-dir "/opt/quantforge/python[hf]"
# compiled kernels and tuning results are cached in /root/.cache/quantforge: mount a volume there to keep them
WORKDIR /workspace
CMD ["quantforge-doctor"]
