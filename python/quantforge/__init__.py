"""QuantForge: MLIR-based INT4 / INT8 / FP8 weight-only GEMM kernels for LLM inference.

    import quantforge.nn as qnn          # quantize_model(model, "int4")           (RTN)
    import quantforge.gptq as qgptq      # quantize_model_gptq(model, calib_ids)   (GPTQ)
    import quantforge.torch_ops          # torch.ops.quantforge.w4a16_gemm

Run ``quantforge-doctor`` (or ``python -m quantforge.doctor``) to check the installation.
"""
__version__ = "0.5.0"
