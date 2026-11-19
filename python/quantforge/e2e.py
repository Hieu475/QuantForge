"""
QuantForge end-to-end harness: qf IR  ->  PTX  ->  run on GPU  ->  compare.

Pipeline (all driven from here):

    quantforge-opt --quantforge-gpu-pipeline   (qf/linalg  -> LLVM + gpu.binary PTX)
    mlir-translate --mlir-to-llvmir            (host code with mgpu* runtime calls)
    llc                                        (host object)
    gcc  + runtime/mgpu_cuda_runtime.c         (shared lib; CUDA driver API)
    ctypes + torch                             (launch + reference check)

Tool discovery
    QUANTFORGE_BUILD_DIR   build tree containing tools/quantforge-opt/quantforge-opt
                           (default: <repo>/build)
    QUANTFORGE_LLVM_BIN    directory with mlir-translate and llc
                           (default: /usr/lib/llvm-18/bin, then PATH)
    CUDA_HOME              CUDA toolkit (headers for the runtime), default /usr/local/cuda

Example
    from quantforge.e2e import W4A16Gemm
    k = W4A16Gemm.build(M=128, K=128, N=128)
    c = k.run(act, packed_w, scale, zero_point)     # torch CUDA tensors
    # any shape (N even): compile for the padded shape, pad/slice at run time
    k = W4A16Gemm.build(M=100, K=70, N=90, pad=True)
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SRC = Path(__file__).resolve().parent / "runtime" / "mgpu_cuda_runtime.c"

# Tile sizes of the pipeline (block-tile-m/n = quantforge-tiling defaults,
# block-tile-k = --quantforge-gpu-pipeline default).
BLOCK_M = BLOCK_N = 128
BLOCK_K = 32


# --------------------------------------------------------------------------- #
# IR generation
# --------------------------------------------------------------------------- #
def tile_sizes(pipeline_opts: str = "") -> tuple[int, int, int]:
    """(block-tile-m, block-tile-n, block-tile-k) selected by a pipeline option string."""
    def get(name: str, default: int) -> int:
        m = re.search(rf"(?:^|\s){name}=(\d+)", pipeline_opts or "")
        return int(m.group(1)) if m else default

    if re.search(r"(?:^|\s)decode=(true|1)", pipeline_opts or ""):
        return get("block-tile-m", 16), get("block-tile-n", 64), get("block-tile-k", 128)
    return get("block-tile-m", BLOCK_M), get("block-tile-n", BLOCK_N), get("block-tile-k", BLOCK_K)


def w4a16_gemm_ir(M: int, K: int, N: int, name: str = "w4a16_gemm", acc: str = "f32",
                  group_size: int | None = None, pipeline_opts: str = "", bits: int = 4) -> str:
    """qf-dialect IR for  C[M,N] += A[M,K](f16) @ dequant(unpack(Wp[K,N/2])).

    `bits`=8: the weights are plain int8 [K,N] (W8A16, no unpack): C += A @ dequant(W).

    `acc` is the accumulator / output element type: "f32" (default, accurate) or "f16".
    `group_size` None: per-tensor scale (f16 scalar) / zero point (i8 scalar).
    `group_size` G: group-wise along K, scale f16[K/G, N] and zero point i8[K/G, N].
    """
    _check_shape(M, K, N, tile_sizes(pipeline_opts))
    if acc not in ("f16", "f32"):
        raise ValueError("acc must be 'f16' or 'f32'")
    if group_size is None:
        st, zt = "tensor<f16>", "tensor<i8>"
    else:
        if group_size <= 0 or K % group_size:
            raise ValueError(f"group_size {group_size} must divide K={K}")
        st, zt = f"tensor<{K // group_size}x{N}xf16>", f"tensor<{K // group_size}x{N}xi8>"
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    wcols = N // 2 if bits == 4 else N
    unpack = (f"%w = qf.unpack %wp : tensor<{K}x{wcols}xi8> -> tensor<{K}x{N}xi8>"
              if bits == 4 else "")
    wsrc = "%w" if bits == 4 else "%wp"
    return f"""func.func @{name}(%act: tensor<{M}x{K}xf16>, %wp: tensor<{K}x{wcols}xi8>,
                      %scale: {st}, %zp: {zt},
                      %c: tensor<{M}x{N}x{acc}>) -> tensor<{M}x{N}x{acc}> {{
  {unpack}
  %wf = qf.dequant {wsrc}, %scale, %zp : tensor<{K}x{N}xi8>, {st}, {zt} -> tensor<{K}x{N}xf16>
  %r = linalg.matmul ins(%act, %wf : tensor<{M}x{K}xf16>, tensor<{K}x{N}xf16>) outs(%c : tensor<{M}x{N}x{acc}>) -> tensor<{M}x{N}x{acc}>
  return %r : tensor<{M}x{N}x{acc}>
}}
"""


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def padded_shape(M: int, K: int, N: int, group_size: int | None = None,
                 pipeline_opts: str = "") -> tuple[int, int, int]:
    """Smallest tileable shape >= (M, K, N) (K also a multiple of the group size)."""
    bm, bn, bk = tile_sizes(pipeline_opts)
    kmul = bk if group_size is None else math.lcm(bk, group_size)
    return _round_up(M, bm), _round_up(K, kmul), _round_up(N, bn)


def _check_shape(M: int, K: int, N: int, tiles: tuple[int, int, int] = (BLOCK_M, BLOCK_N, BLOCK_K)) -> None:
    bm, bn, bk = tiles
    if M % bm or N % bn or K % bk:
        raise ValueError(
            f"shape (M={M}, K={K}, N={N}) must satisfy M%{bm}==0, "
            f"N%{bn}==0, K%{bk}==0 (tiling pass has no remainder handling)"
        )


# --------------------------------------------------------------------------- #
# Tool discovery / process helpers
# --------------------------------------------------------------------------- #
def _find_quantforge_opt() -> str:
    build = Path(os.environ.get("QUANTFORGE_BUILD_DIR", REPO_ROOT / "build"))
    cand = build / "tools" / "quantforge-opt" / "quantforge-opt"
    if cand.exists():
        return str(cand)
    found = shutil.which("quantforge-opt")
    if found:
        return found
    raise FileNotFoundError(
        f"quantforge-opt not found (looked in {cand}); set QUANTFORGE_BUILD_DIR"
    )


def _find_llvm_tool(name: str) -> str:
    for d in filter(None, [os.environ.get("QUANTFORGE_LLVM_BIN"), "/usr/lib/llvm-18/bin"]):
        p = Path(d) / name
        if p.exists():
            return str(p)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} not found; set QUANTFORGE_LLVM_BIN")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if res.returncode != 0:
        err = res.stderr
        if len(err) > 4000:  # diagnostics come first, IR dumps after: keep both ends
            err = err[:2000] + "\n[...]\n" + err[-2000:]
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(cmd)}\n{err}")
    return res


# --------------------------------------------------------------------------- #
# Compilation
# --------------------------------------------------------------------------- #
def extract_ptx(lowered_mlir: str) -> dict[str, str]:
    """Pull the PTX text of every `gpu.binary` out of the lowered module."""
    out: dict[str, str] = {}
    pat = re.compile(r'gpu\.binary @(\w+).*?assembly = "((?:[^"\\]|\\.)*)"', re.S)
    for m in pat.finditer(lowered_mlir):
        raw = re.sub(r"\\([0-9A-Fa-f]{2})", lambda x: chr(int(x.group(1), 16)), m.group(2))
        out[m.group(1)] = raw.rstrip("\x00")
    return out


@dataclass
class Compiled:
    lib_path: Path
    lowered_mlir: Path
    ptx: dict[str, str]
    cached: bool = False  # True: taken from the persistent kernel cache, nothing was compiled


# Persistent cache of compiled kernels: ~/.cache/quantforge/kernels/<key>/ (QUANTFORGE_KERNEL_CACHE
# relocates it, QUANTFORGE_NO_KERNEL_CACHE=1 disables it). The key hashes everything the result
# depends on: the IR, the target and pipeline options, the exact tools (path, size, mtime) and the
# source of the CUDA shim, so a rebuilt compiler never serves a stale kernel.
_CACHED_FILES = ("libqf_kernel.so", "lowered.mlir")


def _kernel_cache_dir() -> Path | None:
    if os.environ.get("QUANTFORGE_NO_KERNEL_CACHE"):
        return None
    return Path(os.environ.get("QUANTFORGE_KERNEL_CACHE",
                               Path.home() / ".cache" / "quantforge" / "kernels"))


def _toolchain_fingerprint() -> str:
    h = hashlib.sha256()
    for tool in (_find_quantforge_opt(), _find_llvm_tool("mlir-translate"), _find_llvm_tool("llc")):
        st = os.stat(tool)
        h.update(f"{os.path.realpath(tool)}:{st.st_size}:{st.st_mtime_ns};".encode())
    h.update(RUNTIME_SRC.read_bytes())
    return h.hexdigest()


def _cache_key(ir: str, chip: str, opts: str) -> str:
    return hashlib.sha256(f"{_toolchain_fingerprint()}\0{chip}\0{opts}\0{ir}".encode()).hexdigest()[:32]


def compile_ir(
    ir: str,
    workdir: str | os.PathLike,
    chip: str = "sm_80",
    pipeline_opts: str = "",
) -> Compiled:
    """Run the full pipeline on `ir` and return the shared library + PTX."""
    opts = f"chip={chip}" + (f" {pipeline_opts}" if pipeline_opts else "")
    cache_root = _kernel_cache_dir()
    entry = None
    if cache_root is not None:
        entry = cache_root / _cache_key(ir, chip, opts)
        if all((entry / f).exists() for f in _CACHED_FILES):
            text = (entry / "lowered.mlir").read_text()
            return Compiled(entry / "libqf_kernel.so", entry / "lowered.mlir", extract_ptx(text), True)
    compiled = _compile_ir_uncached(ir, workdir, chip, opts)
    if entry is not None:
        try:  # publish atomically: concurrent processes see a complete entry or none
            tmp = cache_root / f".tmp-{entry.name}-{os.getpid()}"
            tmp.mkdir(parents=True, exist_ok=True)
            for f in _CACHED_FILES:
                shutil.copy2(Path(workdir) / f, tmp / f)
            try:
                tmp.rename(entry)
            except OSError:  # somebody else published the same entry first
                shutil.rmtree(tmp, ignore_errors=True)
        except OSError:
            pass  # a read-only home must not break compilation
    return compiled


def _compile_ir_uncached(ir: str, workdir, chip: str, opts: str) -> Compiled:
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    src, lowered = work / "input.mlir", work / "lowered.mlir"
    src.write_text(ir)

    _run([_find_quantforge_opt(), str(src), f"--quantforge-gpu-pipeline={opts}", "-o", str(lowered)])

    ll, obj, lib = work / "host.ll", work / "host.o", work / "libqf_kernel.so"
    _run([_find_llvm_tool("mlir-translate"), "--mlir-to-llvmir", str(lowered), "-o", str(ll)])
    _run([_find_llvm_tool("llc"), "-O2", "-relocation-model=pic", "-filetype=obj", str(ll), "-o", str(obj)])

    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    _run(
        [
            "gcc", "-O2", "-fPIC", "-shared", str(obj), str(RUNTIME_SRC), "-o", str(lib),
            f"-I{cuda / 'include'}", "-L/usr/lib/x86_64-linux-gnu", f"-L{cuda / 'lib64'}", "-lcuda",
        ]
    )

    ptx = extract_ptx(lowered.read_text())
    for name, text in ptx.items():
        (work / f"{name}.ptx").write_text(text)
    return Compiled(lib, lowered, ptx)


def ptxas_check(ptx: dict[str, str], arch: str, workdir: str | os.PathLike) -> str:
    """Assemble every PTX module with ptxas; returns the verbose resource report."""
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    ptxas = str(cuda / "bin" / "ptxas")
    report = []
    for name, text in ptx.items():
        p = Path(workdir) / f"{name}.ptx"
        p.write_text(text)
        res = _run([ptxas, f"-arch={arch}", "-v", str(p), "-o", os.devnull])
        report.append(f"[{name}]\n{res.stderr.strip()}")
    return "\n".join(report)


# --------------------------------------------------------------------------- #
# memref descriptor marshalling (MLIR C interface)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _descriptor_type(rank: int):
    class MemRefDesc(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * rank),
            ("strides", ctypes.c_int64 * rank),
        ]

    return MemRefDesc


def _current_raw_stream(torch, device_index: int) -> int:
    """cudaStream_t of PyTorch's current stream (the fast path avoids building a Stream object)."""
    try:
        return torch._C._cuda_getCurrentRawStream(device_index)
    except AttributeError:  # pragma: no cover - older torch
        return torch.cuda.current_stream(device_index).cuda_stream


def _descriptor(t):
    d = _descriptor_type(t.dim())()
    d.allocated = d.aligned = t.data_ptr()
    d.offset = 0
    for i in range(t.dim()):
        d.sizes[i] = t.shape[i]
        d.strides[i] = t.stride(i)
    return d


# --------------------------------------------------------------------------- #
# W4A16 GEMM kernel wrapper
# --------------------------------------------------------------------------- #
class W4A16Gemm:
    """Compiled W4A16 GEMM kernel.

    The compiler requires M, N, K to be multiples of the tile sizes. For other
    shapes build with ``pad=True``: the kernel is compiled for the padded
    shape and `run` zero-pads the inputs and slices the result. Padding is
    exact: padded K columns of the activations are zero (so whatever the
    padded weight nibbles dequantize to is multiplied by 0), and padded M
    rows / N columns are dropped.
    """

    NAME = "w4a16_gemm"  # symbol of the 4-bit kernel; 8-bit kernels are "w8a16_gemm"

    def __init__(self, M: int, K: int, N: int, compiled: Compiled, acc: str = "f32",
                 padded: tuple[int, int, int] | None = None, group_size: int | None = None,
                 bits: int = 4):
        self.M, self.K, self.N, self.acc = M, K, N, acc
        self.group_size = group_size
        self.bits = bits
        self.overwrite = False   # C = A @ W (accumulate=False / swiglu): see build()
        self.swiglu = False      # fused gate/up: output is [M, N/2]
        self.name = f"w{bits}a16_gemm"
        self.Mp, self.Kp, self.Np = padded or (M, K, N)
        self.compiled = compiled
        self._lib = ctypes.CDLL(str(compiled.lib_path))
        fn = getattr(self._lib, f"_mlir_ciface_{self.name}")
        fn.argtypes = [ctypes.c_void_p] * 6
        fn.restype = None
        self._fn = fn
        # the runtime shim launches on the caller's stream (see mgpu_cuda_runtime.c)
        self._set_stream = self._lib.mgpuSetStream
        self._set_stream.argtypes = [ctypes.c_void_p]
        self._set_stream.restype = None
        self._desc_cache = {}
        self._result = _descriptor_type(2)()

    @classmethod
    def build(cls, M: int, K: int, N: int, workdir=None, chip: str = "sm_80",
              acc: str = "f32", pipeline_opts: str = "", pad: bool = False,
              group_size: int | None = None, decode: bool = False,
              bits: int = 4, epilogue: str = "none", accumulate: bool = True) -> "W4A16Gemm":
        """`decode=True` selects the skinny-M schedule (M <= 16 per tile, see PIPELINE.md);
        use it with pad=True for M = 1..16.

        `bits`=8 builds the W8A16 variant: `packed_w` is then a plain int8 [K, N] tensor
        (signed values, zero point / scale as for INT4) and no unpack step runs.

        `epilogue` (relu | silu | gelu): applied to the accumulator before the store, i.e.
        out = act(out_init + A @ W) with out_init = zeros / bias / residual; f32 accumulator only.
        `accumulate=False`: out = A @ W (the kernel never reads `out`: no zeroing needed, and no
        bias / residual). `epilogue="swiglu"` (implies accumulate=False): the fused MLP gate/up:
        the weights hold gate and up interleaved by column (`quant.interleave_gate_up`) and the
        result is silu(A @ W_gate) * (A @ W_up), shape [M, N/2].

        `group_size` None: per-tensor scale / zero point; G: group-wise along K
        (scale f16[K/G, N], zero point i8[K/G, N]; G must divide K)."""
        if epilogue not in ("none", "relu", "silu", "gelu", "swiglu"):
            raise ValueError("epilogue must be one of none | relu | silu | gelu | swiglu")
        if epilogue != "none":
            if acc != "f32":
                raise ValueError("the fused activation epilogue needs acc='f32'")
            pipeline_opts = (pipeline_opts + f" epilogue={epilogue}").strip()
        if epilogue == "swiglu" and (N % 2 or bits != 4):
            raise ValueError("epilogue='swiglu' needs a 4-bit kernel with an even N (gate/up columns interleaved)")
        if not accumulate and epilogue != "swiglu":
            pipeline_opts = (pipeline_opts + " accumulate=false").strip()
        if bits == 4 and N % 2:
            raise ValueError("N must be even (two INT4 weights per byte)")
        if decode and "decode=" not in pipeline_opts:
            pipeline_opts = (pipeline_opts + " decode=true").strip()
        if group_size is not None and K % group_size:
            raise ValueError(f"group_size {group_size} must divide K={K}")
        Mp, Kp, Np = padded_shape(M, K, N, group_size, pipeline_opts) if pad else (M, K, N)
        ir = w4a16_gemm_ir(Mp, Kp, Np, f"w{bits}a16_gemm", acc, group_size, pipeline_opts, bits)
        if workdir is None:
            key = hashlib.sha1(
                f"{Mp}x{Kp}x{Np}:{chip}:{acc}:{pipeline_opts}:g{group_size}:b{bits}".encode()
            ).hexdigest()[:10]
            workdir = Path(tempfile.gettempdir()) / f"quantforge_e2e_{key}"
        kern = cls(M, K, N, compile_ir(ir, workdir, chip, pipeline_opts), acc, (Mp, Kp, Np),
                   group_size, bits)
        kern.overwrite = (not accumulate) or epilogue == "swiglu"
        kern.swiglu = epilogue == "swiglu"
        return kern

    def launch(self, act, packed_w, scale, zero_point, out):
        """Hot path: launch the kernel on already padded, contiguous CUDA tensors and return
        immediately (asynchronous, on PyTorch's current stream). Shapes must be exactly the
        compiled ones -- act [Mp, Kp], packed_w [Kp, Np/2] (or [Kp, Np] for 8 bit), out [Mp, Np] in
        the accumulator dtype, scale / zero_point [Kp/G, Np] (or scalars when built per tensor) --
        and 16-byte aligned. Nothing is checked or allocated here; `run` is the safe wrapper.
        Memref descriptors are cached per (pointer, shape)."""
        import torch

        cache = self._desc_cache
        args = []
        for t in (act, packed_w, scale, zero_point, out):
            key = (t.data_ptr(), tuple(t.shape))
            d = cache.get(key)
            if d is None:
                if len(cache) > 8192:
                    cache.clear()
                d = cache[key] = _descriptor(t)
            args.append(ctypes.byref(d))
        self._set_stream(_current_raw_stream(torch, act.device.index or 0))
        self._fn(ctypes.byref(self._result), *args)
        return out

    def _run_overwrite(self, act, packed_w, scale, zero_point, out, bias, residual):
        """`run` for kernels that never read their output (accumulate=False / swiglu)."""
        import torch
        import torch.nn.functional as F

        if bias is not None or residual is not None:
            raise ValueError("bias / residual need an accumulating kernel (accumulate=True)")
        torch.cuda.init()
        dtype = torch.float32 if self.acc == "f32" else torch.float16
        cols = self.N // 2 if self.swiglu else self.N
        if out is not None:
            assert out.is_cuda and out.is_contiguous() and tuple(out.shape) == (self.M, cols), \
                f"out must be a contiguous CUDA tensor of shape {(self.M, cols)}"
        scale, zero_point = scale.contiguous(), zero_point.contiguous()
        for t in (act, packed_w, scale, zero_point):
            assert t.is_cuda and t.is_contiguous(), "all tensors must be contiguous CUDA tensors"
            if t.dim() and t.data_ptr() % 16:
                raise ValueError("tensors must be 16-byte aligned (got an interior view?)")
        assert tuple(act.shape) == (self.M, self.K), (act.shape, (self.M, self.K))
        wcols = self.N // 2 if self.bits == 4 else self.N
        assert tuple(packed_w.shape) == (self.K, wcols), packed_w.shape

        pad_m, pad_k, pad_n = self.Mp != self.M, self.Kp != self.K, self.Np != self.N
        run_act, run_w = act, packed_w
        if pad_m or pad_k:
            run_act = F.pad(act, (0, self.Kp - self.K, 0, self.Mp - self.M)).contiguous()
        if pad_k or pad_n:
            run_w = F.pad(packed_w, (0, (self.Np - self.N) // (2 if self.bits == 4 else 1), 0,
                                     self.Kp - self.K)).contiguous()
        if self.group_size is None:
            scale, zero_point = scale.reshape(()), zero_point.reshape(())
        else:
            g = (self.K // self.group_size, self.N)
            assert tuple(scale.shape) == g and tuple(zero_point.shape) == g, (scale.shape, g)
            if pad_k or pad_n:
                pad = (0, self.Np - self.N, 0, (self.Kp - self.K) // self.group_size)
                scale = F.pad(scale, pad).contiguous()
                zero_point = F.pad(zero_point, pad).contiguous()
        run_out = torch.empty(self.Mp, self.Np, dtype=dtype, device=act.device)  # fully written
        self._set_stream(_current_raw_stream(torch, act.device.index or 0))
        args = [_descriptor(t) for t in (run_act, run_w, scale, zero_point, run_out)]
        self._fn(ctypes.byref(self._result), *[ctypes.byref(a) for a in args])
        res = run_out[: self.M, :cols]
        if out is None:
            return res if res.is_contiguous() else res.contiguous()
        out.copy_(res)
        return out

    def run(self, act, packed_w, scale, zero_point, out=None, bias=None, residual=None):
        """act[M,K] f16, packed_w[K,N/2] int8, scale f16, zero_point int8.

        scale / zero_point are scalars (per-tensor build) or [K/G, N] tensors (group-wise).

        Computes out += act @ dequant(unpack(packed_w)); `out` (f32 or f16 per `acc`)
        defaults to zeros. All tensors must be contiguous CUDA tensors.

        Epilogue by accumulator initialisation: the kernel *accumulates* into `out`, so
        `bias` ([N]) and `residual` ([M, N]) are fused simply by starting from them
        (out = residual + bias; out += A @ W) -- no extra pass over the result. Only
        valid when `out` is not given. An activation can be fused with `build(epilogue=...)`.
        """
        import torch
        import torch.nn.functional as F

        if self.overwrite:
            return self._run_overwrite(act, packed_w, scale, zero_point, out, bias, residual)
        torch.cuda.init()
        dtype = torch.float32 if self.acc == "f32" else torch.float16
        padded = (self.Mp, self.Kp, self.Np) != (self.M, self.K, self.N)
        if out is None:
            if residual is not None:
                assert tuple(residual.shape) == (self.M, self.N), residual.shape
                out = residual.to(dtype).contiguous().clone()
            else:
                out = torch.zeros(self.M, self.N, dtype=dtype, device=act.device)
            if bias is not None:
                assert tuple(bias.shape) == (self.N,), bias.shape
                out += bias.to(dtype)
        elif bias is not None or residual is not None:
            raise ValueError("bias / residual are only applied when `out` is not given")
        scale, zero_point = scale.contiguous(), zero_point.contiguous()
        for t in (act, packed_w, scale, zero_point, out):
            assert t.is_cuda and t.is_contiguous(), "all tensors must be contiguous CUDA tensors"
        # the generated kernels issue 16-byte vector loads / cp.async on these
        for name, t in (("act", act), ("packed_w", packed_w), ("scale", scale),
                        ("zero_point", zero_point), ("out", out)):
            if t.dim() and t.data_ptr() % 16:
                raise ValueError(f"{name} must be 16-byte aligned (got an interior view?)")
        assert tuple(act.shape) == (self.M, self.K), (act.shape, (self.M, self.K))
        wcols = self.N // 2 if self.bits == 4 else self.N
        assert tuple(packed_w.shape) == (self.K, wcols), packed_w.shape
        assert tuple(out.shape) == (self.M, self.N), out.shape

        # pad only what the padding actually changes (an M-only pad must not copy the weights)
        pad_m, pad_k, pad_n = self.Mp != self.M, self.Kp != self.K, self.Np != self.N
        run_act, run_w, run_out = act, packed_w, out
        if pad_m or pad_k:
            run_act = F.pad(act, (0, self.Kp - self.K, 0, self.Mp - self.M)).contiguous()
        if pad_k or pad_n:
            run_w = F.pad(packed_w, (0, (self.Np - self.N) // (2 if self.bits == 4 else 1), 0,
                                     self.Kp - self.K)).contiguous()
        if pad_m or pad_n:
            run_out = F.pad(out, (0, self.Np - self.N, 0, self.Mp - self.M)).contiguous()

        if self.group_size is None:
            scale = scale.reshape(())
            zero_point = zero_point.reshape(())
        else:
            g = (self.K // self.group_size, self.N)
            assert tuple(scale.shape) == g, f"scale must be {g}, got {tuple(scale.shape)}"
            assert tuple(zero_point.shape) == g, f"zero_point must be {g}"
            if pad_k or pad_n:  # padded K rows / N columns only meet zero activations / dropped outputs
                pad = (0, self.Np - self.N, 0, (self.Kp - self.K) // self.group_size)
                scale = F.pad(scale, pad).contiguous()
                zero_point = F.pad(zero_point, pad).contiguous()
        args = [_descriptor(t) for t in (run_act, run_w, scale, zero_point, run_out)]
        result = _descriptor_type(2)()
        # launch on PyTorch's current stream: ordered after the caller's pending work and
        # (the generated host code waits for its stream) complete when this returns
        self._set_stream(_current_raw_stream(torch, act.device.index or 0))
        self._fn(ctypes.byref(result), *[ctypes.byref(a) for a in args])
        if padded:
            out.copy_(run_out[: self.M, : self.N])
        return out


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #
def pack_int4(q):
    """q: int8 tensor [K,N] with values 0..15  ->  packed [K,N/2] (low nibble = even column)."""
    import torch

    return ((q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)).to(torch.int8)


def dequant_reference(q, scale, zero_point):
    """f16 weights (q - zp) * scale; q is the unpacked [K,N] 0..15 tensor.

    scale / zero_point: scalars (per-tensor) or [K/G, N] (group-wise along K)."""
    import torch

    if scale.dim() == 0 or scale.numel() == 1:
        w = (q.float() - float(zero_point)) * float(scale)
    else:
        G = q.shape[0] // scale.shape[0]
        s = scale.float().repeat_interleave(G, dim=0)
        z = zero_point.float().repeat_interleave(G, dim=0)
        w = (q.float() - z) * s
    return w.to(torch.float16)


def reference_gemm(act, q, scale, zero_point):
    """fp32 reference of act @ ((q - zp) * scale), q holds the *unpacked* 0..15 values."""
    return act.float() @ dequant_reference(q, scale, zero_point).float()
