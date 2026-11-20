"""Do vLLM's Marlin kernels and the QuantForge kernels (quantization="quantforge") produce the same text?

    python benchmarks/vllm_compare_outputs.py Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4

Runs the same GPTQ checkpoint with both methods (two engine launches), greedy decoding of 64 tokens for a few prompts, and compares
the generated tokens and the log-probability of every generated token (max / mean absolute difference).
"""
import json
import subprocess
import sys

PROMPTS = ["Explain in one paragraph why the sky is blue.", "Write a Python function that reverses a linked list.",
           "What is the capital of France, and what is it famous for?", "Translate to German: The weather is nice today.",
           "List three prime numbers larger than 100."]


def run(model, quant, out):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, quantization=quant, dtype="float16", gpu_memory_utilization=0.5, max_model_len=512,
              enable_prefix_caching=False)
    res = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=64, logprobs=0))
    data = [{"tokens": list(r.outputs[0].token_ids),
             "logprobs": [next(iter(lp.values())).logprob for lp in r.outputs[0].logprobs]} for r in res]
    json.dump(data, open(out, "w"))


def main():
    model = sys.argv[1]
    if len(sys.argv) > 3 and sys.argv[2] == "--child":
        run(model, sys.argv[3], sys.argv[4]); return
    outs = {}
    for q in ("gptq_marlin", "quantforge"):
        outs[q] = f"/tmp/vllm_cmp_{q}.json"
        subprocess.run([sys.executable, __file__, model, "--child", q, outs[q]], check=True, capture_output=True)
    a, b = (json.load(open(outs[q])) for q in ("gptq_marlin", "quantforge"))
    same = 0; total = 0; diffs = []
    for x, y in zip(a, b):
        n = min(len(x["tokens"]), len(y["tokens"]))
        k = next((i for i in range(n) if x["tokens"][i] != y["tokens"][i]), n)      # length of the common prefix
        same += k; total += n
        diffs += [abs(p - q) for p, q in zip(x["logprobs"][:k], y["logprobs"][:k])]
        print(f"  prompt: common prefix {k}/{n} tokens")
    print(f"identical prefix tokens: {same}/{total}; |logprob difference| on them: max {max(diffs):.4f}, mean {sum(diffs) / len(diffs):.5f}")


if __name__ == "__main__":
    main()
