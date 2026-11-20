"""vLLM end-to-end batch-1 decode speed (offline engine, CUDA graphs, 256 new tokens, best of 3).

    python benchmarks/vllm_e2e.py Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 gptq_marlin     # Marlin kernels
    python benchmarks/vllm_e2e.py Qwen/Qwen2.5-0.5B-Instruct none                       # fp16
"""
import sys, time, warnings
warnings.filterwarnings("ignore")


def main():
    from vllm import LLM, SamplingParams
    name = sys.argv[1]
    quant = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "none" else None
    llm = LLM(model=name, quantization=quant, dtype="float16", gpu_memory_utilization=0.55, max_model_len=1024,
              enable_prefix_caching=False)
    sp = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True)
    prompt = "Explain in one paragraph why the sky is blue."
    llm.generate([prompt], sp)          # warm-up
    best = 0
    for _ in range(3):
        t = time.time(); out = llm.generate([prompt], sp); dt = time.time() - t
        best = max(best, len(out[0].outputs[0].token_ids) / dt)
    print(f"RESULT {name} quant={quant}: {best:.1f} tokens/s (batch 1, 256 new tokens)")


if __name__ == "__main__":
    main()
