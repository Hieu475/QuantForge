"""vLLM plugin: the `quantforge` quantization method is registered; a real model runs if enabled.

    QUANTFORGE_SLOW_TESTS=1 python -m pytest test/E2E/test_vllm_plugin.py -q     # needs vLLM, a GPU and the model in the HF cache
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4"


def test_quantforge_quantization_method_is_registered():
    pytest.importorskip("vllm")
    import quantforge.vllm_plugin as vp
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

    vp.register()
    cfg = get_quantization_config("quantforge")
    assert cfg is vp.QuantForgeConfig and issubclass(cfg, AutoGPTQConfig) and cfg.get_name() == "quantforge"
    assert cfg.override_quantization_method({"quant_method": "gptq"}, "quantforge") == "quantforge"
    assert cfg.override_quantization_method({"quant_method": "gptq"}, None) is None      # only on request
    assert cfg.override_quantization_method({"quant_method": "awq"}, "quantforge") is None


@pytest.mark.skipif(os.environ.get("QUANTFORGE_SLOW_TESTS") != "1", reason="slow test: set QUANTFORGE_SLOW_TESTS=1")
def test_vllm_generates_the_same_text_as_marlin_for_a_simple_prompt():
    pytest.importorskip("vllm")
    import subprocess

    code = (
        "import sys\nfrom vllm import LLM, SamplingParams\n"
        f"llm = LLM('{MODEL}', quantization=sys.argv[1], dtype='float16', gpu_memory_utilization=0.5, max_model_len=256)\n"
        "out = llm.generate(['What is the capital of France?'], SamplingParams(temperature=0.0, max_tokens=24))\n"
        "print('TEXT:' + out[0].outputs[0].text.replace(chr(10), ' '))\n")
    texts = {}
    for q in ("gptq_marlin", "quantforge"):
        r = subprocess.run([sys.executable, "-c", code, q], capture_output=True, text=True)
        if "TEXT:" not in r.stdout:
            pytest.skip(f"vLLM could not run {MODEL} ({r.stderr[-200:]})")
        texts[q] = r.stdout.split("TEXT:")[1].strip()
    assert "Paris" in texts["quantforge"]
    assert texts["quantforge"][:40] == texts["gptq_marlin"][:40], texts
