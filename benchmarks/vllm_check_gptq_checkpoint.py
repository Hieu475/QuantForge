"""Does vLLM + the QuantForge kernels run a GPTQ checkpoint written by `quantforge.gptq_checkpoint.save_gptq` like Hugging Face runs the same weights?

    python benchmarks/vllm_check_gptq_checkpoint.py <checkpoint dir> <the fp16 model it was made from>

Reference: the fp16 Hugging Face model whose decoder weights are replaced by the dequantised weights read back from the checkpoint tensors; both greedy-decode
64 tokens of a few prompts; prints how many tokens agree and the difference of the log-probabilities of the generated tokens.
"""
import json
import subprocess
import sys

PROMPTS = ["Explain in one paragraph why the sky is blue.", "Write a Python function that reverses a linked list.",
           "What is the capital of France, and what is it famous for?", "Translate to German: The weather is nice today.",
           "List three prime numbers larger than 100."]


def vllm_side(ckpt, out):
    from vllm import LLM, SamplingParams
    llm = LLM(model=ckpt, quantization="quantforge", dtype="float16", gpu_memory_utilization=0.5, max_model_len=512, enable_prefix_caching=False)
    res = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=64, logprobs=0))
    json.dump([{"tokens": list(r.outputs[0].token_ids),
                "logprobs": [next(iter(lp.values())).logprob for lp in r.outputs[0].logprobs]} for r in res], open(out, "w"))


def hf_side(ckpt, base):
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quantforge import quant

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16).cuda().eval()
    G = json.load(open(f"{ckpt}/config.json"))["quantization_config"]["group_size"]
    with safe_open(f"{ckpt}/model.safetensors", "pt", device="cuda") as f:
        for name, mod in model.named_modules():
            if f"{name}.qweight" in f.keys():
                packed, scale, zero = quant.from_gptq(f.get_tensor(f"{name}.qweight"), f.get_tensor(f"{name}.qzeros"), f.get_tensor(f"{name}.scales"))
                q = quant.unpack_int4(packed).float()
                w = ((q.view(-1, G, q.shape[1]) - zero.float().unsqueeze(1)) * scale.float().unsqueeze(1)).reshape(q.shape).t()
                mod.weight.data = w.half().contiguous()
    out = []
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].cuda() \
            if False else tok(p, return_tensors="pt").input_ids.cuda()
        g = model.generate(ids, max_new_tokens=64, do_sample=False, output_scores=True, return_dict_in_generate=True)
        new = g.sequences[0, ids.shape[1]:].tolist()
        lps = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(g.scores, new)]
        out.append({"tokens": new, "logprobs": lps})
    return out


def main():
    ckpt, base = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 4 and sys.argv[3] == "--child":
        return vllm_side(ckpt, sys.argv[4])
    out = "/tmp/vllm_gptq_ckpt.json"
    subprocess.run([sys.executable, __file__, ckpt, base, "--child", out], check=True, capture_output=True)
    a = json.load(open(out))
    b = hf_side(ckpt, base)
    for i, (x, y) in enumerate(zip(a, b)):
        n = next((k for k, (s, t) in enumerate(zip(x["tokens"], y["tokens"])) if s != t), min(len(x["tokens"]), len(y["tokens"])))
        d = [abs(s - t) for s, t in zip(x["logprobs"][:n], y["logprobs"][:n])]
        print(f"prompt {i}: {n} of {len(y['tokens'])} tokens identical; mean |logprob difference| over them {sum(d) / max(len(d), 1):.4f}, max {max(d, default=0):.4f}")


if __name__ == "__main__":
    main()
