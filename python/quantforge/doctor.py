"""``quantforge-doctor``: check that this machine can generate and run QuantForge kernels.

Checks the Python / torch / transformers versions, the CUDA GPU, nvcc / ptxas, the `quantforge-opt` tool (and that
it can lower a small GEMM), and runs one tiny GEMM against a PyTorch reference. Exit code 0 = everything works.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys


def _line(ok: bool | None, what: str, detail: str = "") -> bool:
    mark = {True: "ok  ", False: "FAIL", None: "info"}[ok]
    print(f"  [{mark}] {what}" + (f": {detail}" if detail else ""))
    return ok is not False


def _check_pack_kernel(packs) -> bool:
    """Without the compiler: run the probe shape recorded in a pack's meta.json from the pack and compare it with a reference."""
    import json

    import torch

    from . import torch_ops
    from .e2e import dequant_reference, pack_int4

    for d in packs:
        try:
            K, N, G = json.loads((d / "meta.json").read_text())["probe"]
        except (OSError, ValueError, KeyError):
            continue
        g = torch.Generator(device="cuda").manual_seed(0)
        q = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int8, generator=g)
        scale = (torch.rand(K // G, N, device="cuda", generator=g) * 0.1 + 0.01).half()
        zp = torch.randint(0, 16, (K // G, N), device="cuda", dtype=torch.int8, generator=g)
        deq = dequant_reference(q, scale, zp).float()
        before, worst = torch_ops.DENSE_FALLBACKS, 0.0
        for M in (1, 7, 16, 64):
            x = torch.randn(M, K, device="cuda", generator=g).half()
            ref = x.float() @ deq
            worst = max(worst, ((torch_ops.w4a16_linear(x, pack_int4(q), scale, zp, G).float() - ref).abs().max() / ref.abs().max()).item())
        dense = torch_ops.DENSE_FALLBACKS - before
        ok = worst < 5e-3 and not dense
        return _line(ok, "kernel from pack", f"{K}x{N} at 1/7/16/64 rows, max relative error {worst:.1e}, C++ runtime {torch_ops.CPP_RUNTIME}"
                     + (f", {dense} dense fallback calls (the pack lacks this shape)" if dense else ""))
    return _line(None, "kernel from pack", "no pack has a probe shape (built before `meta.json` recorded one)")


def main() -> int:
    from . import __version__

    print(f"quantforge {__version__}")
    good = True
    good &= _line(sys.version_info >= (3, 9), "python", sys.version.split()[0])
    try:
        import torch

        good &= _line(True, "torch", f"{torch.__version__}, CUDA {torch.version.cuda}")
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            cap = torch.cuda.get_device_capability(0)
            good &= _line(cap >= (8, 0), "GPU", f"{torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]}; needs sm_80+)")
        else:
            good &= _line(False, "GPU", "no CUDA device visible to torch")
    except Exception as e:  # noqa: BLE001
        good &= _line(False, "torch", f"{type(e).__name__}: {e}")
        has_gpu = False
    for mod in ("transformers", "safetensors"):
        try:
            _line(None, mod, importlib.import_module(mod).__version__)
        except Exception:  # noqa: BLE001
            _line(None, mod, "not installed (only needed for quantforge.nn / gptq with Hugging Face models)")
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    for tool in ("nvcc", "ptxas"):
        found = shutil.which(tool) or (os.path.join(cuda, "bin", tool) if os.path.exists(os.path.join(cuda, "bin", tool)) else None)
        _line(found is not None, tool, found or f"not on PATH and not in {cuda}/bin")
    packs = []
    try:
        from .e2e import ext_tag, pack_dirs

        packs = pack_dirs()
        for d in packs:
            n = len(list((d / "kernels").glob("*.so")))
            ext = (d / "ext" / ext_tag() / "quantforge_rt.so").exists()
            _line(None, "kernel pack", f"{d}: {n} kernels, C++ runtime for this torch: {'yes' if ext else 'no'}")
    except Exception as e:  # noqa: BLE001
        _line(None, "kernel pack", f"{type(e).__name__}: {e}")
    try:
        from .e2e import _find_quantforge_opt

        opt = _find_quantforge_opt()
        good &= _line(True, "quantforge-opt", opt)
    except Exception as e:  # noqa: BLE001
        # without the compiler the kernels must come from a pack: not a failure when one is installed
        good &= _line(None if packs else False, "quantforge-opt",
                      f"{e} (build it with CMake and set QUANTFORGE_BUILD_DIR" + (")" if not packs else "; kernels come from the pack)"))
        opt = None
    if opt and has_gpu:
        try:
            import torch

            from .e2e import W4A16Gemm, dequant_reference, pack_int4

            M, K, N, G = 16, 256, 128, 64
            g = torch.Generator(device="cuda").manual_seed(0)
            act = torch.randn(M, K, device="cuda", generator=g).half()
            q = torch.randint(0, 16, (K, N), device="cuda", dtype=torch.int8, generator=g)
            scale = (torch.rand(K // G, N, device="cuda", generator=g) * 0.1 + 0.01).half()
            zp = torch.randint(0, 16, (K // G, N), device="cuda", dtype=torch.int8, generator=g)
            kern = W4A16Gemm.build(M, K, N, group_size=G, pad=True)
            out = kern.run(act, pack_int4(q), scale, zp)
            ref = act.float() @ dequant_reference(q, scale, zp).float()
            err = (out.float() - ref).abs().max().item()
            good &= _line(err < 2e-3 * ref.abs().max().item() + 1e-2, "kernel", f"compiled, ran, max|err| {err:.4f}")
        except Exception as e:  # noqa: BLE001
            good &= _line(False, "kernel", f"{type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}")
    if not opt and packs and has_gpu:
        good &= _check_pack_kernel(packs)
    print("everything works" if good else "problems found (see above)")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
