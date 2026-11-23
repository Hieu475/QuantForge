"""Pure decode step time of a MoE checkpoint in vLLM (the prefill and the first tokens cancel by differencing two generation lengths; the KV cache must not
limit the batch, or preemption is measured instead of kernels).

    python benchmarks/vllm_moe_decode.py <checkpoint dir> <none|quantforge>        # none = vLLM's own kernels (Marlin MoE for symmetric GPTQ)
"""
from pathlib import Path
import os, sys, time, warnings
warnings.filterwarnings("ignore")
def main():
    from vllm import LLM, SamplingParams
    model, quant = sys.argv[1], sys.argv[2]
    quant = None if quant == "none" else quant
    llm = LLM(model, quantization=quant, dtype="float16", gpu_memory_utilization=0.9, max_model_len=256, max_num_seqs=32, max_num_batched_tokens=1024, enforce_eager=bool(os.environ.get("EAGER")))
    base = "Explain how a transformer language model works and why attention matters for long documents. "
    for n in (1, 8, 16, 32):
        prompts = [base + f"({i}) " for i in range(n)]
        llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True))
        def run(mt):
            best = 1e9
            for _ in range(3):
                t = time.time(); llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=mt, ignore_eos=True)); best = min(best, time.time() - t)
            return best
        a, b = run(12), run(60)
        print(f"DECODE reqs={n:3d}: {n * 48 / (b - a):8.1f} tok/s (step {1000 * (b - a) / 48:.2f} ms)", flush=True)
if __name__ == "__main__":
    main()
