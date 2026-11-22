"""Kernel packs: ready-made kernels for the layer shapes of a model, so that a machine without LLVM, ``quantforge-opt`` or ``nvcc`` runs them.

    python -m quantforge.pack build --out ./pack --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4     # needs the toolchain once
    python -m quantforge.pack check ./pack --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4           # runs every row count with JIT off
    QUANTFORGE_PACK=./pack python your_vllm_script.py                                             # no toolchain needed

A pack is a directory: ``kernels/<key>.so`` (one self-contained kernel library each, linked only against libcuda; the key hashes the target, the
pipeline options and the IR, see ``e2e.pack_key``), ``plans.json`` (the measured decisions: K-fold factors, atomic split-K plans), ``meta.json``.
Building runs the layers of the model through exactly the path the vLLM plugin takes (``kfold.linear``) for every row count the op serves, with
``QUANTFORGE_PACK_RECORD`` set, which copies each kernel the process uses. ``check`` then runs the same sweep with ``QUANTFORGE_NO_JIT=1``: it fails
if any shape would need the dense fallback.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROWS = tuple(range(1, 17)) + (17, 24, 32, 33, 48, 64, 65, 96, 128, 129, 192, 256, 257, 384, 385, 512, 513, 768, 769, 1024, 1025, 1536, 1537,
                              2048, 2049, 3000, 4096, 8192)


def load_config(model: str) -> dict:
    p = Path(model)
    if p.is_dir():
        p = p / "config.json"
    if p.is_file():
        return json.loads(p.read_text())
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(model, "config.json")).read_text())


def layer_shapes(cfg: dict, tp: int = 1) -> list[tuple[str, int, int, bool]]:
    """``(name, K, N, has_bias)`` of the four GEMMs of a decoder layer, per tensor-parallel rank (fused qkv and gate/up, as vLLM runs them)."""
    cfg = cfg.get("text_config", cfg)
    h = cfg["hidden_size"]
    nh = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads", nh)
    hd = cfg.get("head_dim") or h // nh
    inter = cfg["intermediate_size"]
    mt = cfg.get("model_type", "")
    qkv_bias = bool(cfg.get("attention_bias", mt.startswith("qwen2")))
    out = [("qkv", h, (nh + 2 * nkv) * hd // tp, qkv_bias),
           ("o", nh * hd // tp, h, False),
           ("gate_up", h, 2 * inter // tp, bool(cfg.get("mlp_bias", False))),
           ("down", inter // tp, h, bool(cfg.get("mlp_bias", False)))]
    return out


def sweep_layers(shapes, group: int = 128, rows=ROWS, log=print) -> None:
    """Run each layer through the op for every row count (what the plugin does at run time)."""
    import torch

    from . import kfold
    from .quant import pack_int4

    dev = torch.device("cuda")
    for name, K, N, bias in shapes:
        if K % group or N % 2:
            log(f"  {name}: K={K} N={N} not servable with group {group}, skipped")
            continue
        t0 = time.time()
        g = torch.Generator(device=dev).manual_seed(0)
        q = torch.randint(0, 16, (K, N), device=dev, dtype=torch.int8, generator=g)
        packed = pack_int4(q)
        scale = (torch.rand(K // group, N, device=dev, generator=g) * 0.1 + 0.01).half()
        zero = torch.randint(0, 16, (K // group, N), device=dev, dtype=torch.int8, generator=g)
        S = kfold.plan_ksplit(K, N, group, 4)
        folded = kfold.fold_k(packed, scale, zero, S) if S > 1 else kfold.no_fold(dev)
        b = (torch.rand(N, device=dev, generator=g) - 0.5).half() if bias else None
        for M in rows:
            x = (torch.rand(M, K, device=dev, generator=g) * 2 - 1).half()
            kfold.linear(x, packed, scale, zero, folded, group, b, None)
        torch.cuda.synchronize()
        log(f"  {name:8s} K={K:6d} N={N:6d} bias={int(bias)} fold={S}: {time.time() - t0:5.1f} s")


def _pack_plans_from_cache(dest: Path, shapes, group: int) -> int:
    """Copy the planner decisions (fold factors, atomic plans) of these shapes from the autotune cache into the pack."""
    from . import autotune as at

    cache = at._load()
    keep = {}
    for _, K, N, _b in shapes:
        for k, v in cache.items():
            if k.startswith(("kfold", "atomic")) and f"|g{group}|" in k and k.endswith(f"|K{K}|N{N}"):
                keep[k] = v
    old = {}
    if (dest / "plans.json").exists():
        old = json.loads((dest / "plans.json").read_text())
    old.update(keep)
    (dest / "plans.json").write_text(json.dumps(old, indent=1, sort_keys=True) + "\n")
    return len(keep)


def cmd_build(a) -> None:
    out = Path(a.out).resolve()
    (out / "kernels").mkdir(parents=True, exist_ok=True)
    os.environ["QUANTFORGE_PACK_RECORD"] = str(out)
    import torch

    from . import torch_ops  # noqa: F401 - registers the op

    models = []
    shapes = []
    for m in a.model:
        cfg = load_config(m)
        for tp in a.tp:
            sh = layer_shapes(cfg, tp)
            print(f"{m} (tp={tp}): " + ", ".join(f"{n} {K}x{N}" for n, K, N, _ in sh), flush=True)
            shapes += sh
        models.append(m)
    seen, uniq = set(), []
    for sh in shapes:
        if sh[1:] not in seen:
            seen.add(sh[1:])
            uniq.append(sh)
    t0 = time.time()
    sweep_layers(uniq, a.group, log=lambda s: print(s, flush=True))
    n = _pack_plans_from_cache(out, uniq, a.group)
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"models": []}
    meta["models"] = sorted(set(meta["models"]) | set(models))
    meta["gpu"] = torch.cuda.get_device_name(0)
    meta["torch"] = torch.__version__
    meta["abi"] = 1
    meta["kernels"] = len(list((out / "kernels").glob("*.so")))
    meta_p.write_text(json.dumps(meta, indent=1) + "\n")
    size = sum(f.stat().st_size for f in (out / "kernels").glob("*.so")) / 2 ** 20
    print(f"pack {out}: {meta['kernels']} kernels, {size:.1f} MiB, {n} plans, {time.time() - t0:.0f} s")


def cmd_check(a) -> None:
    env = dict(os.environ, QUANTFORGE_NO_JIT="1", QUANTFORGE_PACK=str(Path(a.pack).resolve()))
    env.pop("QUANTFORGE_PACK_RECORD", None)
    env["QUANTFORGE_KERNEL_CACHE"] = "/nonexistent"
    code = (
        "import sys, json; from quantforge import pack, torch_ops\n"
        "shapes=[]\n"
        f"for m in {a.model!r}:\n"
        f"    for tp in {a.tp!r}:\n"
        "        shapes += pack.layer_shapes(pack.load_config(m), tp)\n"
        f"pack.sweep_layers(shapes, {a.group}, rows=tuple(range(1, {a.max_rows + 1})) + pack.ROWS)\n"
        "print('DENSE', torch_ops.DENSE_FALLBACKS)\n"
        "sys.exit(1 if torch_ops.DENSE_FALLBACKS else 0)\n")
    r = subprocess.run([sys.executable, "-c", code], env=env)
    print("pack covers the model: no dense fallback" if r.returncode == 0 else "pack does NOT cover every shape (see the warning above)")
    sys.exit(r.returncode)


def cmd_ext(a) -> None:
    """Build the C++ torch runtime once for this torch / CUDA / Python and store it in the pack (hosts without nvcc / a compiler then load it)."""
    import shutil
    import tempfile

    import torch
    from torch.utils import cpp_extension

    from .e2e import ext_tag

    out = Path(a.out).resolve() / "ext" / ext_tag()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", a.arch)
    rt = Path(__file__).resolve().parent / "runtime"
    cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    with tempfile.TemporaryDirectory() as tmp:
        cpp_extension.load(name="quantforge_rt", sources=[str(rt / "torch_runtime.cpp"), str(rt / "kfold_reduce.cu")], with_cuda=True,
                           extra_include_paths=[f"{cuda}/include"], build_directory=tmp, verbose=False)
        shutil.copy2(Path(tmp) / "quantforge_rt.so", out / "quantforge_rt.so")
    print(f"{out / 'quantforge_rt.so'}  (TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}, torch {torch.__version__})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m quantforge.pack", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="compile the kernels of the given models into a pack (needs the toolchain)")
    b.add_argument("--out", required=True)
    b.add_argument("--model", action="append", required=True, help="Hugging Face id, local directory or config.json (repeatable)")
    b.add_argument("--tp", type=int, nargs="+", default=[1], help="tensor-parallel sizes (default 1)")
    b.add_argument("--group", type=int, default=128)
    e = sub.add_parser("ext", help="build the C++ torch runtime for this torch into the pack")
    e.add_argument("--out", required=True)
    e.add_argument("--arch", default="8.0;8.6;8.9+PTX", help="TORCH_CUDA_ARCH_LIST of the one CUDA kernel of the runtime")
    c = sub.add_parser("check", help="run every row count with JIT off and fail on a dense fallback")
    c.add_argument("pack")
    c.add_argument("--model", action="append", required=True)
    c.add_argument("--tp", type=int, nargs="+", default=[1])
    c.add_argument("--group", type=int, default=128)
    c.add_argument("--max-rows", type=int, default=300)
    a = ap.parse_args(argv)
    {"build": cmd_build, "check": cmd_check, "ext": cmd_ext}[a.cmd](a)


if __name__ == "__main__":
    main()
