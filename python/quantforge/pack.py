"""Kernel packs: ready-made kernels for the layer shapes of a model, so that a machine without LLVM, ``quantforge-opt`` or ``nvcc`` runs them.

    python -m quantforge.pack build --out ./pack --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4     # needs the toolchain once
    python -m quantforge.pack archive ./pack pack.tar.gz ; python -m quantforge.pack install pack.tar.gz   # ship it / unpack it where it is used
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

ROWS = tuple(range(1, 17)) + (17, 24, 32, 33, 48, 64, 65, 80, 96, 97, 128, 129, 160, 161, 192, 193, 224, 225, 256, 257, 384, 385, 512, 513, 768, 769, 1024, 1025, 1536, 1537,
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

    from . import atomic_plan, kfold
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
        atomic_plan.preplan(K, N, group)
        folded = kfold.fold_k(packed, scale, zero, S) if S > 1 else kfold.no_fold(dev)
        b = (torch.rand(N, device=dev, generator=g) - 0.5).half() if bias else None
        for M in rows:
            x = (torch.rand(M, K, device=dev, generator=g) * 2 - 1).half()
            kfold.linear(x, packed, scale, zero, folded, group, b, None)
        torch.cuda.synchronize()
        log(f"  {name:8s} K={K:6d} N={N:6d} bias={int(bias)} fold={S}: {time.time() - t0:5.1f} s")


def sweep_w4a8(shapes, rows=ROWS, log=print, modes=("dual", "compact")) -> None:
    """The W4A8 hybrid op (`w4a8_op`, `quantization="quantforge_w4a8"`) for the same layers: both weight layouts, every row count."""
    import torch

    from . import kfold, quant, w4a8_op

    dev = torch.device("cuda")
    for name, K, N, bias in shapes:
        if K % 128 or N % 128:
            log(f"  {name}: K={K} N={N} is not a multiple of 128, no W4A8 layer")
            continue
        t0 = time.time()
        g = torch.Generator(device=dev).manual_seed(0)
        parts = [quant.w4a8_quantize((torch.randn(min(2048, N - r), K, device=dev, generator=g) * 0.03)) for r in range(0, N, 2048)]
        packed, gs, zp, sw = (torch.cat([p[i] for p in parts]) for i in range(4))
        b = (torch.rand(N, device=dev, generator=g) - 0.5).half() if bias else None
        for mode in modes:
            if mode == "compact":
                e = kfold.no_fold(dev)
                p16, s16, z16, pf, sf, zf = e[0], e[1], e[2], e[0], e[1], e[2]
                scale_nk = (gs.float() * sw.float()[:, None]).half()
            else:
                p16, s16, z16 = w4a8_op.from_w4a8(packed, gs, zp, sw.float())
                S = kfold.plan_ksplit(K, N, 128, 4)
                pf, sf, zf = kfold.fold_k(p16, s16, z16, S) if S > 1 else kfold.no_fold(dev)
                scale_nk = torch.empty(0, device=dev, dtype=torch.float16)
            for M in rows:
                x = (torch.rand(M, K, device=dev, generator=g) * 2 - 1).half()
                torch.ops.quantforge.w4a8_hybrid(x, packed, gs, zp, sw.float(), b, p16, s16, z16, pf, sf, zf, 128, scale_nk)
        torch.cuda.synchronize()
        log(f"  w4a8 {name:8s} K={K:6d} N={N:6d}: {time.time() - t0:5.1f} s")


def _pack_plans_from_cache(dest: Path, shapes, group: int) -> int:
    """Copy the planner decisions (fold factors, atomic plans) of these shapes from the autotune cache into the pack."""
    from . import autotune as at

    cache = at._load()
    keep = {}
    for _, K, N, _b in shapes:
        for k, v in cache.items():
            if k.startswith(("kfold", "atomic")) and f"|g{group}|" in k and k.endswith(f"|K{K}|N{N}"):
                keep[k] = v
            elif k.startswith("tile") and k.endswith(f"|K{K}|N{N}"):       # per-shape tile schedules (group independent)
                keep[k] = v
    old = {}
    if (dest / "plans.json").exists():
        old = json.loads((dest / "plans.json").read_text())
    old.update(keep)
    (dest / "plans.json").write_text(json.dumps(old, indent=1, sort_keys=True) + "\n")
    return len(keep)


def cmd_record(a) -> None:
    """Second pass of ``build`` (a fresh process): every plan is cached by now, so nothing is measured and every kernel the op asks for is recorded."""
    os.environ["QUANTFORGE_PACK_RECORD"] = str(Path(a.out).resolve())
    from . import torch_ops  # noqa: F401

    sweep_layers([tuple(x) for x in json.loads(a.shapes)], a.group, log=lambda s: print(s, flush=True))
    if a.w4a8:
        sweep_w4a8([tuple(x) for x in json.loads(a.shapes)], log=lambda s: print(s, flush=True))


def cmd_build(a) -> None:
    out = Path(a.out).resolve()
    (out / "kernels").mkdir(parents=True, exist_ok=True)
    os.environ.pop("QUANTFORGE_PACK_RECORD", None)
    os.environ["QUANTFORGE_PLAN_MID"] = "1"          # plan the 96..256-row schedules too (they are looked up, never measured, at run time)
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
    # pass 1 measures the planners' candidates (K-fold factors, atomic split-K) and caches the decisions. The kernels the planners build live on in
    # this process, so a recording made here would miss them: pass 2 runs in a fresh process with every plan cached and records everything.
    print("pass 1: plans", flush=True)
    sweep_layers(uniq, a.group, log=lambda s: print(s, flush=True))
    print("pass 2: recording kernels", flush=True)
    r = subprocess.run([sys.executable, "-m", "quantforge.pack", "_record", "--out", str(out), "--group", str(a.group), "--shapes", json.dumps(uniq)] + (["--w4a8"] if a.w4a8 else []))
    if r.returncode:
        raise SystemExit(r.returncode)
    n = _pack_plans_from_cache(out, uniq, a.group)
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"models": []}
    meta["models"] = sorted(set(meta["models"]) | set(models))
    meta["gpu"] = torch.cuda.get_device_name(0)
    meta["torch"] = torch.__version__
    meta["abi"] = 1
    small = min((sh for sh in uniq if sh[1] % a.group == 0), key=lambda sh: sh[1] * sh[2], default=None)
    if small and "probe" not in meta:
        meta["probe"] = [small[1], small[2], a.group]       # `quantforge-doctor` runs this shape from the pack
    meta["kernels"] = len(list((out / "kernels").glob("*.so")))
    meta_p.write_text(json.dumps(meta, indent=1) + "\n")
    size = sum(f.stat().st_size for f in (out / "kernels").glob("*.so")) / 2 ** 20
    print(f"pack {out}: {meta['kernels']} kernels, {size:.1f} MiB, {n} plans, {time.time() - t0:.0f} s")


def cmd_check(a) -> None:
    env = dict(os.environ, QUANTFORGE_NO_JIT="1", QUANTFORGE_PACK=str(Path(a.pack).resolve()))
    env.pop("QUANTFORGE_PACK_RECORD", None)
    env["QUANTFORGE_KERNEL_CACHE"] = "/nonexistent"
    env["QUANTFORGE_TUNE_CACHE"] = "/nonexistent/tune.json"      # decisions must come from the pack, not from this machine's cache
    code = (
        "import sys, json; from quantforge import pack, torch_ops\n"
        "shapes=[]\n"
        f"for m in {a.model!r}:\n"
        f"    for tp in {a.tp!r}:\n"
        "        shapes += pack.layer_shapes(pack.load_config(m), tp)\n"
        f"pack.sweep_layers(shapes, {a.group}, rows=tuple(range(1, {a.max_rows + 1})) + pack.ROWS)\n"
        f"if {a.w4a8!r}:\n"
        "    from quantforge import w4a8_op\n"
        f"    pack.sweep_w4a8(shapes, rows=tuple(range(1, {a.max_rows + 1})) + pack.ROWS)\n"
        "    torch_ops.DENSE_FALLBACKS += w4a8_op.FALLBACKS\n"
        "print('DENSE', torch_ops.DENSE_FALLBACKS, 'MISSING KERNELS', len(torch_ops.PACK_MISSES))\n"
        "for m in sorted(set(torch_ops.PACK_MISSES))[:20]: print('  missing (M, K, N, kind):', m)\n"
        "sys.exit(1 if torch_ops.DENSE_FALLBACKS or torch_ops.PACK_MISSES else 0)\n")
    r = subprocess.run([sys.executable, "-c", code], env=env)
    print("pack covers the model: every kernel is in the pack, no dense fallback" if r.returncode == 0 else "pack does NOT cover every shape (see above)")
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


def cmd_archive(a) -> None:
    import tarfile

    src = Path(a.pack).resolve()
    with tarfile.open(a.out, "w:gz") as tf:
        for sub in ("kernels", "ext", "plans.json", "meta.json"):
            if (src / sub).exists():
                tf.add(src / sub, arcname=sub)
    print(f"{a.out}: {Path(a.out).stat().st_size / 2 ** 20:.1f} MiB")


def cmd_install(a) -> None:
    """Unpack a pack archive (a file or an http(s) URL) into ``~/.cache/quantforge/pack``, merging with what is there."""
    import shutil
    import tarfile
    import tempfile
    import urllib.request

    dest = Path(a.dest).expanduser() if a.dest else Path.home() / ".cache" / "quantforge" / "pack"
    with tempfile.TemporaryDirectory() as tmp:
        arch = a.source
        if a.source.startswith(("http://", "https://")):
            arch = str(Path(tmp) / "pack.tar.gz")
            urllib.request.urlretrieve(a.source, arch)
        stage = Path(tmp) / "x"
        with tarfile.open(arch) as tf:
            for m in tf.getmembers():       # no absolute paths, no escaping the directory, no links
                if m.name.startswith(("/", "..")) or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                    raise SystemExit(f"refusing archive member {m.name!r}")
            tf.extractall(stage, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        for f in stage.rglob("*"):
            if f.is_file():
                t = dest / f.relative_to(stage)
                if f.name == "plans.json" and t.exists():
                    merged = json.loads(t.read_text())
                    merged.update(json.loads(f.read_text()))
                    t.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
                elif f.name != "meta.json" or not t.exists():
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, t)
    n = len(list((dest / "kernels").glob("*.so")))
    print(f"installed into {dest}: {n} kernels in total")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m quantforge.pack", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="compile the kernels of the given models into a pack (needs the toolchain)")
    b.add_argument("--out", required=True)
    b.add_argument("--model", action="append", required=True, help="Hugging Face id, local directory or config.json (repeatable)")
    b.add_argument("--tp", type=int, nargs="+", default=[1], help="tensor-parallel sizes (default 1)")
    b.add_argument("--group", type=int, default=128)
    b.add_argument("--w4a8", action="store_true", help="also record the kernels of the W4A8 hybrid op (`quantization=\"quantforge_w4a8\"`)")
    ar = sub.add_parser("archive", help="pack a pack directory into a .tar.gz")
    ar.add_argument("pack")
    ar.add_argument("out")
    ins = sub.add_parser("install", help="unpack a pack archive (file or URL) into ~/.cache/quantforge/pack")
    ins.add_argument("source")
    ins.add_argument("--dest", help="another directory (then point QUANTFORGE_PACK at it)")
    rc = sub.add_parser("_record", help=argparse.SUPPRESS)
    rc.add_argument("--out", required=True)
    rc.add_argument("--group", type=int, default=128)
    rc.add_argument("--shapes", required=True)
    rc.add_argument("--w4a8", action="store_true")
    e = sub.add_parser("ext", help="build the C++ torch runtime for this torch into the pack")
    e.add_argument("--out", required=True)
    e.add_argument("--arch", default="8.0;8.6;8.9+PTX", help="TORCH_CUDA_ARCH_LIST of the one CUDA kernel of the runtime")
    c = sub.add_parser("check", help="run every row count with JIT off and fail on a dense fallback")
    c.add_argument("pack")
    c.add_argument("--model", action="append", required=True)
    c.add_argument("--tp", type=int, nargs="+", default=[1])
    c.add_argument("--group", type=int, default=128)
    c.add_argument("--max-rows", type=int, default=300)
    c.add_argument("--w4a8", action="store_true", help="check the W4A8 hybrid op too")
    a = ap.parse_args(argv)
    {"build": cmd_build, "check": cmd_check, "ext": cmd_ext, "_record": cmd_record, "archive": cmd_archive, "install": cmd_install}[a.cmd](a)


if __name__ == "__main__":
    main()
