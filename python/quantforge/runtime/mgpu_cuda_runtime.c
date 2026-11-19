/*===----------------------------------------------------------------------===//
 * Minimal CUDA driver-API runtime for MLIR `gpu.launch_func` host code.
 *
 * MLIR 18's LLVM IR translation of gpu.launch_func emits calls to a small set
 * of `mgpu*` entry points (normally provided by libmlir_cuda_runtime, which the
 * distro LLVM packages do not ship). This file implements exactly that set on
 * top of the CUDA driver API so QuantForge can run its generated PTX without
 * building MLIR from source.
 *
 * Context handling: the runtime uses whatever CUDA context is current on the
 * calling thread (e.g. the one PyTorch created). If there is none, it retains
 * the device-0 primary context.
 *
 * Build:  gcc -O2 -fPIC -shared mgpu_cuda_runtime.c -o libmgpu.so \
 *             -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -lcuda
 *===----------------------------------------------------------------------===*/

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    CUresult result_ = (expr);                                                 \
    if (result_ != CUDA_SUCCESS) {                                             \
      const char *name_ = "?", *msg_ = "?";                                    \
      cuGetErrorName(result_, &name_);                                         \
      cuGetErrorString(result_, &msg_);                                        \
      fprintf(stderr, "[mgpu] %s:%d: %s failed: %s (%s)\n", __FILE__,          \
              __LINE__, #expr, name_, msg_);                                   \
      abort();                                                                 \
    }                                                                          \
  } while (0)

static void ensureContext(void) {
  CUcontext ctx = NULL;
  CUDA_CHECK(cuCtxGetCurrent(&ctx));
  if (ctx)
    return;
  CUDA_CHECK(cuInit(0));
  CUdevice dev;
  CUDA_CHECK(cuDeviceGet(&dev, 0));
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CUDA_CHECK(cuCtxSetCurrent(ctx));
}

#define EXPORT __attribute__((visibility("default")))

static __thread CUstream g_externalStream = NULL;
static __thread int g_haveExternalStream = 0;

/* The launching thread's stream. When the host program (the Python harness)
 * sets one with mgpuSetStream -- PyTorch's current stream -- every
 * `gpu.launch_func` of the generated host code runs on it: no stream is
 * created / destroyed per call and the kernel is ordered after the caller's
 * pending work (out.zero_(), the previous layer, ...) without a device-wide
 * synchronisation. */

EXPORT void mgpuSetStream(CUstream stream) {
  g_externalStream = stream;
  g_haveExternalStream = 1;
}

EXPORT CUstream mgpuStreamCreate(void) {
  ensureContext();
  if (g_haveExternalStream)
    return g_externalStream;
  CUstream stream = NULL;
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  return stream;
}

EXPORT void mgpuStreamDestroy(CUstream stream) {
  if (g_haveExternalStream && stream == g_externalStream)
    return; /* not ours */
  CUDA_CHECK(cuStreamDestroy(stream));
}

EXPORT void mgpuStreamSynchronize(CUstream stream) {
  /* On the caller's stream the launch is asynchronous like any other PyTorch op:
   * later work on that stream is ordered after it, and the caller synchronises
   * when it needs the result (tensor.item(), .cpu(), torch.cuda.synchronize()).
   * Blocking the host here would serialise CPU launch work with the GPU. */
  if (g_haveExternalStream && stream == g_externalStream)
    return;
  CUDA_CHECK(cuStreamSynchronize(stream));
}

/* Modules are cached by the address of their (immutable, embedded) PTX image:
 * the JIT result is reused by every later call instead of re-running
 * cuModuleLoadDataEx (~100 us+) per launch. They live until process exit, so
 * mgpuModuleUnload is a no-op for cached modules. */
#define MODULE_CACHE_SIZE 64
static struct {
  const void *data;
  int optLevel;
  CUmodule module;
} g_modules[MODULE_CACHE_SIZE];
static int g_numModules = 0;

static CUmodule cachedModule(const void *data, int optLevel) {
  for (int i = 0; i < g_numModules; ++i)
    if (g_modules[i].data == data && g_modules[i].optLevel == optLevel)
      return g_modules[i].module;
  return NULL;
}

static int isCached(CUmodule module) {
  for (int i = 0; i < g_numModules; ++i)
    if (g_modules[i].module == module)
      return 1;
  return 0;
}

/* PTX text (NUL-terminated) is JIT-compiled by the driver for the current
 * device. optLevel comes from the nvvm target's `O` option. */
EXPORT CUmodule mgpuModuleLoadJIT(void *data, int optLevel) {
  ensureContext();
  CUmodule hit = cachedModule(data, optLevel);
  if (hit)
    return hit;
  CUmodule module = NULL;
  char errorLog[4096] = {0};
  unsigned int level = optLevel < 0 ? 0 : (optLevel > 4 ? 4 : optLevel);
  CUjit_option options[] = {CU_JIT_ERROR_LOG_BUFFER,
                            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                            CU_JIT_OPTIMIZATION_LEVEL};
  void *values[] = {errorLog, (void *)(uintptr_t)sizeof(errorLog),
                    (void *)(uintptr_t)level};
  CUresult result = cuModuleLoadDataEx(&module, data, 3, options, values);
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "[mgpu] cuModuleLoadDataEx failed (%d): %s\n", (int)result,
            errorLog);
    abort();
  }
  if (g_numModules < MODULE_CACHE_SIZE) {
    g_modules[g_numModules].data = data;
    g_modules[g_numModules].optLevel = optLevel;
    g_modules[g_numModules].module = module;
    ++g_numModules;
  }
  return module;
}

EXPORT CUmodule mgpuModuleLoad(void *data, size_t gpuBlobSize) {
  (void)gpuBlobSize;
  ensureContext();
  CUmodule module = NULL;
  CUDA_CHECK(cuModuleLoadData(&module, data));
  return module;
}

EXPORT void mgpuModuleUnload(CUmodule module) {
  if (isCached(module))
    return;
  CUDA_CHECK(cuModuleUnload(module));
}

EXPORT CUfunction mgpuModuleGetFunction(CUmodule module, const char *name) {
  CUfunction function = NULL;
  CUDA_CHECK(cuModuleGetFunction(&function, module, name));
  return function;
}

EXPORT void mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                             intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                             intptr_t blockZ, int32_t smem, CUstream stream,
                             void **params, void **extra, size_t paramsCount) {
  (void)paramsCount;
  if (smem > 48 * 1024)
    CUDA_CHECK(cuFuncSetAttribute(
        function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
  CUDA_CHECK(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY,
                            blockZ, smem, stream, params, extra));
}

EXPORT void *mgpuMemAlloc(uint64_t sizeBytes, CUstream stream,
                          int8_t isHostShared) {
  (void)stream;
  ensureContext();
  CUdeviceptr ptr = 0;
  if (sizeBytes == 0)
    return NULL;
  if (isHostShared)
    CUDA_CHECK(cuMemAllocManaged(&ptr, sizeBytes, CU_MEM_ATTACH_GLOBAL));
  else
    CUDA_CHECK(cuMemAlloc(&ptr, sizeBytes));
  return (void *)ptr;
}

EXPORT void mgpuMemFree(void *ptr, CUstream stream) {
  (void)stream;
  if (ptr)
    CUDA_CHECK(cuMemFree((CUdeviceptr)ptr));
}
