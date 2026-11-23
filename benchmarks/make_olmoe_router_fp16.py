"""Make a copy of `shuyuej/OLMoE-1B-7B-0924-GPTQ` that vLLM can load: that checkpoint quantises the router (`mlp.gate`) too, which vLLM's OLMoE does not expect; the router is
dequantised back to an fp16 `weight`.

    python benchmarks/make_olmoe_router_fp16.py        # downloads the checkpoint (3.7 GiB) if needed, writes ./olmoe_gptq_fixed (or the directory given as argument)
"""
import sys, shutil, glob, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from quantforge import quant
from huggingface_hub import snapshot_download
src = Path(snapshot_download("shuyuej/OLMoE-1B-7B-0924-GPTQ"))
dst = Path(sys.argv[1] if len(sys.argv) > 1 else "olmoe_gptq_fixed")
dst.mkdir(exist_ok=True)
tensors = {}
with safe_open(glob.glob(str(src / "*.safetensors"))[0], "pt", device="cpu") as f:
    keys = list(f.keys())
    gates = {k[:-len(".qweight")] for k in keys if k.endswith("mlp.gate.qweight")}
    for k in keys:
        base = k.rsplit(".", 1)[0]
        if base in gates:
            continue
        tensors[k] = f.get_tensor(k)
    for g in sorted(gates):
        qw, qz, sc = (f.get_tensor(f"{g}.{n}").cuda() for n in ("qweight", "qzeros", "scales"))
        packed, scale, zero = quant.from_gptq(qw, qz, sc, None, 1)
        q = quant.unpack_int4(packed).float()
        gk = q.shape[0] // scale.shape[0]
        w = (q - zero.float().repeat_interleave(gk, 0)) * scale.float().repeat_interleave(gk, 0)     # [K, N]
        tensors[f"{g}.weight"] = w.t().contiguous().half().cpu()
print("router layers repaired:", len(gates), "tensors:", len(tensors))
save_file(tensors, str(dst / "model.safetensors"), metadata={"format": "pt"})
for n in src.iterdir():
    if n.suffix != ".safetensors" and n.name != "model.safetensors.index.json" and n.is_file():
        shutil.copy2(n, dst / n.name)
print(json.load(open(dst / "config.json")).get("quantization_config", {}).get("modules_in_block_to_quantize"))
