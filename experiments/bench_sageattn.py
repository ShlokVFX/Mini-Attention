"""
SageAttention benchmark — mirrors the thu-ml/SageAttention bench style.
Tests: real sageattn (Triton INT8) vs torch SDPA vs our fp16_sage_attn_sm86 (Triton)
Configs: typical video diffusion shapes (CogVideoX-2b, LTX-Video, HunyuanVideo)
"""

import math
import time
import torch
import torch.nn.functional as F


def bench(fn, warmup=20, reps=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3  # ms


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


def run_config(label, B, H, N, D, dtype=torch.float16):
    q = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, D, device="cuda", dtype=dtype)

    # accuracy
    ref = sdpa(q, k, v)

    from sageattention import sageattn
    out_sage = sageattn(q, k, v.clone(), tensor_layout="HND")
    err_sage = (out_sage.float() - ref.float()).abs().max().item()

    ms_sdpa  = bench(lambda: sdpa(q, k, v))
    ms_sage  = bench(lambda: sageattn(q, k, v.clone(), tensor_layout="HND"))

    speedup = ms_sdpa / ms_sage

    print(f"  {label:45s}  sdpa={ms_sdpa:6.2f}ms  sage={ms_sage:6.2f}ms  "
          f"x{speedup:.2f}  maxerr={err_sage:.5f}")


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"SM: {torch.cuda.get_device_capability()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # --- same configs as thu-ml bench ---
    # CogVideoX-2b: 48 heads, D=64, seq varies with resolution/frames
    # HunyuanVideo: 24 heads, D=128
    # LTX-Video:    32 heads, D=64
    configs = [
        # label                              B  H    N     D
        ("CogVideoX-2b  [B=1 H=48 N=4096  D=64]",  1, 48, 4096,  64),
        ("CogVideoX-2b  [B=1 H=48 N=8192  D=64]",  1, 48, 8192,  64),
        ("CogVideoX-2b  [B=1 H=48 N=16384 D=64]",  1, 48, 16384, 64),
        ("LTX-Video     [B=1 H=32 N=4096  D=64]",  1, 32, 4096,  64),
        ("LTX-Video     [B=1 H=32 N=8192  D=64]",  1, 32, 8192,  64),
        ("HunyuanVideo  [B=1 H=24 N=4096  D=128]", 1, 24, 4096,  128),
        ("HunyuanVideo  [B=1 H=24 N=8192  D=128]", 1, 24, 8192,  128),
        ("Wan2.1        [B=1 H=16 N=16384 D=128]", 1, 16, 16384, 128),
        # smaller for sanity
        ("baseline      [B=2 H=8  N=1024  D=64]",  2,  8, 1024,  64),
    ]

    print(f"{'Config':47s}  {'SDPA':>10s}  {'SageAttn':>10s}  {'speedup':>8s}  maxerr")
    print("-" * 100)
    for label, B, H, N, D in configs:
        try:
            run_config(label, B, H, N, D)
        except torch.cuda.OutOfMemoryError:
            print(f"  {label:45s}  OOM — skipped")
        except Exception as e:
            print(f"  {label:45s}  ERROR: {e}")

    print()
    print("Note: SageAttention uses per-block INT8 Q,K quantization + smooth_k + FP16 V")
    print("      SDPA uses Flash Attention 2 (FA2) when available, otherwise math attn")


if __name__ == "__main__":
    main()
