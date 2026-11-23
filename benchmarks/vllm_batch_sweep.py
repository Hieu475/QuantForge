"""vLLM decode throughput versus the number of concurrent requests: Marlin against the QuantForge plugin.

    python benchmarks/vllm_batch_sweep.py Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 gptq_marlin   # or: quantforge
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def main():
    from vllm import LLM, SamplingParams
    name, quant = sys.argv[1], sys.argv[2]
    batches = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else "1,4,16,64".split(","))]
    llm = LLM(model=name, quantization=quant, dtype="float16", gpu_memory_utilization=float(__import__("os").environ.get("QF_GPU_UTIL", "0.55")), max_model_len=1024,
              enable_prefix_caching=False, kv_cache_dtype=__import__("os").environ.get("QF_KV", "auto"))
    sp = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
    prompts = [f"Explain topic number {i} of physics in one paragraph." for i in range(max(batches))]
    llm.generate(prompts[:max(batches)], sp)                     # warm-up (compiles every kernel bucket)
    out = []
    for b in batches:
        best = 0.0
        for _ in range(3):
            t = time.time(); res = llm.generate(prompts[:b], sp); dt = time.time() - t
            best = max(best, sum(len(r.outputs[0].token_ids) for r in res) / dt)
        out.append((b, best))
    print("RESULT " + quant + " " + " ".join(f"{b}:{t:.0f}" for b, t in out))


if __name__ == "__main__":
    main()
