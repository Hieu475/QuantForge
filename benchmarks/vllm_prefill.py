"""vLLM prefill and decode throughput of one (model, quantization) pair.

    python benchmarks/vllm_prefill.py Qwen/Qwen2.5-0.5B-Instruct none                  # fp16
    python benchmarks/vllm_prefill.py Qwen/Qwen2.5-0.5B-Instruct quantforge_w4a8       # online W4A8
    python benchmarks/vllm_prefill.py Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 gptq_marlin # or: quantforge

Prefill: 16 prompts of about 1000 tokens, one generated token each (prompt tokens per second, best of 3). Decode: 1 and 16 requests of
a short prompt, 128 generated tokens (generated tokens per second, best of 3). Prints one RESULT line.
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def main():
    from vllm import LLM, SamplingParams

    name, quant = sys.argv[1], sys.argv[2]
    llm = LLM(model=name, quantization=None if quant == "none" else quant, dtype="float16", gpu_memory_utilization=0.6,
              max_model_len=2048, max_num_batched_tokens=4096, enable_prefix_caching=False)
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    long_prompts = [" ".join(words[(i + j) % len(words)] for j in range(900)) for i in range(16)]
    ntok = len(llm.get_tokenizer().encode(long_prompts[0]))
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    spd = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
    short = [f"Explain topic number {i} of physics in one paragraph." for i in range(16)]
    llm.generate(long_prompts[:4], sp1)
    llm.generate(short, spd)                                       # warm-up: compiles every kernel bucket
    best = 0.0
    for _ in range(3):
        t = time.time(); llm.generate(long_prompts, sp1); best = max(best, 16 * ntok / (time.time() - t))
    out = [f"prefill:{best:.0f}"]
    for b in (1, 16):
        bd = 0.0
        for _ in range(3):
            t = time.time(); res = llm.generate(short[:b], spd); dt = time.time() - t
            bd = max(bd, sum(len(r.outputs[0].token_ids) for r in res) / dt)
        out.append(f"decode{b}:{bd:.0f}")
    print(f"RESULT {name} {quant} prompt_tokens={ntok} " + " ".join(out))


if __name__ == "__main__":
    main()
