"""
QuantForge Phase 1: PyTorch Baseline for INT4 GEMV

Ground truth implementation for W4A16 unpacking and GEMV.
This provides the golden tensor and latency baseline to compare
against the MLIR-compiled kernel.

Usage:
    conda activate quantforge
    python python/quantforge/baseline.py
"""

import torch
import time


def pack_int4(weights: torch.Tensor) -> torch.Tensor:
    """Pack two INT4 values into one INT8 byte.
    
    Args:
        weights: tensor of shape (M, N) with values in [0, 15]
    Returns:
        packed: tensor of shape (M, N//2) as int8
    """
    assert weights.shape[-1] % 2 == 0, "Last dim must be even"
    assert weights.dtype == torch.int8

    low = weights[..., 0::2] & 0x0F    # Even indices → low nibble
    high = weights[..., 1::2] & 0x0F   # Odd indices → high nibble
    packed = low | (high << 4)
    return packed.to(torch.int8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack INT8 tensor containing two INT4 values per byte.
    
    Args:
        packed: tensor of shape (M, N//2) as int8
    Returns:
        unpacked: tensor of shape (M, N) as int8
    """
    low = packed & 0x0F                 # Low nibble: bits [3:0]
    high = (packed >> 4) & 0x0F         # High nibble: bits [7:4]

    # Interleave: [low0, high0, low1, high1, ...]
    M = packed.shape[0]
    N_half = packed.shape[1]
    unpacked = torch.stack([low, high], dim=-1).reshape(M, N_half * 2)
    return unpacked


def gemv_int4_baseline(x: torch.Tensor, w_packed: torch.Tensor,
                       scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """W4A16 GEMV: Y = X @ dequant(unpack(W_packed))^T
    
    Args:
        x: activation tensor (1, K) in FP16
        w_packed: packed INT4 weights (N, K//2) in INT8
        scale: per-channel scale (N,) in FP16
        zero_point: per-channel zero point (N,) in INT8
    Returns:
        y: output tensor (1, N) in FP16
    """
    # Step 1: Unpack INT4 → INT8
    w_unpacked = unpack_int4(w_packed)  # (N, K)

    # Step 2: Dequantize to FP16
    w_fp16 = (w_unpacked.to(torch.float16) - zero_point.unsqueeze(1).to(torch.float16)) * scale.unsqueeze(1)

    # Step 3: GEMV
    y = x @ w_fp16.T
    return y


def benchmark(fn, *args, warmup=10, repeat=100):
    """Benchmark a CUDA function with proper synchronization."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Timed runs
    start = time.time()
    for _ in range(repeat):
        fn(*args)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / repeat
    return elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dimensions matching a typical LLM layer
    M = 1       # Batch size (decode phase = 1 token)
    K = 4096    # Input features (hidden dim)
    N = 4096    # Output features

    print(f"GEMV: ({M}x{K}) @ ({N}x{K})^T → ({M}x{N})")
    print(f"W4A16: weights stored as INT4, activations as FP16\n")

    # Create random test data
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.float16, device=device)

    # Random INT4 weights [0, 15]
    w_full = torch.randint(0, 16, (N, K), dtype=torch.int8, device=device)
    w_packed = pack_int4(w_full)  # (N, K//2)

    scale = torch.randn(N, dtype=torch.float16, device=device) * 0.01
    zero_point = torch.randint(0, 16, (N,), dtype=torch.int8, device=device)

    # Verify correctness: pack → unpack should be identity
    w_roundtrip = unpack_int4(w_packed)
    assert torch.equal(w_full, w_roundtrip), "Pack/Unpack roundtrip failed!"
    print("✅ Pack/Unpack roundtrip: PASS")

    # Compute golden tensor
    golden = gemv_int4_baseline(x, w_packed, scale, zero_point)
    print(f"✅ Golden tensor shape: {golden.shape}, dtype: {golden.dtype}")

    # Save golden tensor for future comparison
    torch.save({
        'x': x.cpu(),
        'w_packed': w_packed.cpu(),
        'scale': scale.cpu(),
        'zero_point': zero_point.cpu(),
        'golden': golden.cpu(),
    }, 'test/golden_tensors.pt')
    print("✅ Golden tensors saved to test/golden_tensors.pt")

    # Benchmark
    latency = benchmark(gemv_int4_baseline, x, w_packed, scale, zero_point)
    print(f"\n📊 PyTorch Naive W4A16 GEMV Latency: {latency * 1000:.3f} ms")
    print(f"   (This is the baseline to beat with MLIR compiler)")

    # Memory bandwidth analysis
    w_bytes = w_packed.numel() * 1  # INT8 = 1 byte each
    x_bytes = x.numel() * 2        # FP16 = 2 bytes each
    total_bytes = w_bytes + x_bytes
    bw_achieved = total_bytes / latency / 1e9  # GB/s
    print(f"\n📊 Effective Bandwidth: {bw_achieved:.1f} GB/s")
    print(f"   (RTX 4050 theoretical peak: ~192 GB/s)")
    print(f"   Utilization: {bw_achieved / 192 * 100:.1f}%")


if __name__ == "__main__":
    main()
