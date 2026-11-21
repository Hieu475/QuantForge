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
OTHER_FORMATS = [("Qwen/Qwen2.5-0.5B-Instruct-AWQ", "awq_marlin"), ("Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8", "gptq_marlin")]


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
    assert cfg.override_quantization_method({"quant_method": "fp8"}, "quantforge") is None        # formats it cannot read


def test_config_reads_gptq_8bit_and_awq_checkpoints():
    pytest.importorskip("vllm")
    import quantforge.vllm_plugin as vp

    cfg = vp.QuantForgeConfig
    assert cfg.override_quantization_method({"quant_method": "awq"}, "quantforge") == "quantforge"
    assert cfg.override_quantization_method({"quant_method": "awq"}, None) is None
    g8 = cfg.from_config({"quant_method": "gptq", "bits": 8, "group_size": 128, "desc_act": False, "sym": True})
    assert g8.weight_bits == 8 and not g8.awq
    awq = cfg.from_config({"quant_method": "awq", "w_bit": 4, "q_group_size": 128, "zero_point": True,
                           "modules_to_not_convert": ["visual"]})
    assert awq.awq and awq.pack_factor == 8 and awq.group_size == 128 and awq.modules_to_not_convert == ["visual"]
    with pytest.raises(NotImplementedError):                                  # AWQ without zero points: not supported
        cfg.from_config({"quant_method": "awq", "w_bit": 4, "q_group_size": 128, "zero_point": False})


@pytest.mark.skipif(os.environ.get("QUANTFORGE_SLOW_TESTS") != "1", reason="slow test: set QUANTFORGE_SLOW_TESTS=1")
@pytest.mark.parametrize("model,reference", [(MODEL, "gptq_marlin")] + OTHER_FORMATS)
def test_vllm_generates_the_same_text_as_marlin_for_a_simple_prompt(model, reference):
    pytest.importorskip("vllm")
    import subprocess

    code = (
        "import sys\nfrom vllm import LLM, SamplingParams\n"
        f"llm = LLM('{model}', quantization=sys.argv[1], dtype='float16', gpu_memory_utilization=0.5, max_model_len=256)\n"
        "out = llm.generate(['What is the capital of France?'], SamplingParams(temperature=0.0, max_tokens=24))\n"
        "print('TEXT:' + out[0].outputs[0].text.replace(chr(10), ' '))\n")
    texts = {}
    for q in (reference, "quantforge"):
        r = subprocess.run([sys.executable, "-c", code, q], capture_output=True, text=True)
        if "TEXT:" not in r.stdout:
            if q == reference:       # the reference itself cannot run here (no model, no memory): nothing to compare
                pytest.skip(f"vLLM could not run {model} with {q} ({r.stderr[-200:]})")
            pytest.fail(f"quantforge failed where {reference} worked on {model}:\n{r.stderr[-1500:]}")   # never hide a plugin bug
        texts[q] = r.stdout.split("TEXT:")[1].strip()
    assert "Paris" in texts["quantforge"]
    assert texts["quantforge"][:40] == texts[reference][:40], texts


def test_register_never_raises_even_without_a_usable_vllm(monkeypatch):
    """vLLM calls the entry point on every start: with a missing or too old vLLM it may only warn."""
    import warnings

    import quantforge.vllm_plugin as vp

    monkeypatch.setattr(vp, "HAVE_VLLM", False)
    monkeypatch.setattr(vp, "_IMPORT_ERROR", ImportError("cannot import name 'AutoGPTQConfig'"), raising=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vp.register()
    assert any("quantforge" in str(x.message) and vp.MIN_VLLM in str(x.message) for x in w)
