#!/usr/bin/env python3
"""
Benchmark: src_1-7_simplified optimized for RTX 5090 (Blackwell SM120).

Architecture notes:
  - SM120 (Blackwell): 100 KB smem/SM, 256-byte L2 sectors, 1.8 TB/s bandwidth
  - Compile with sm_120 native code + sm_90 PTX fallback
  - New Bc=128 tile configs improve arithmetic intensity ~33% over Bc=64

Usage:
    uv run python kernels/flash_attn/cuda/_bench_sm120.py
"""
import os, statistics, sys
from pathlib import Path

# Override arch list for SM120 (RTX 5090 Blackwell)
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_HERE  = Path(__file__).resolve().parent
_KDIR  = _HERE / "kernels"
sys.path.insert(0, str(_HERE / "py"))
from flash_helpers.kernel_configs import get_kernel_progression_configs, FlashForwardKernelConfig, DType

# --------------------------------------------------------------------------- #
# NVCC flags: target SM120 natively + SM90 PTX as forward-compatible fallback.
# Key SM120 changes vs SM86:
#   • 100 KB shared memory per SM  (vs 64 KB on SM86)
#   • 256-byte L2 cache sectors    (prefetch hint updated in ptx_functions.cuh)
#   • Higher warp-level throughput for mma.sync
# --------------------------------------------------------------------------- #
NVCC_FLAGS = [
    "-std=c++20",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    # Native SM120 binary code (RTX 5090 Blackwell)
    "-gencode", "arch=compute_120,code=sm_120",
    # SM90 PTX fallback for JIT compilation on future archs
    "-gencode", "arch=compute_90,code=compute_90",
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

def compile_k(tag, src_dir):
    d = _HERE / "build" / tag
    d.mkdir(parents=True, exist_ok=True)
    return load(
        name=tag,
        sources=[str(src_dir / "flash_attention.cu")],
        extra_include_paths=[str(src_dir / "include")],
        extra_cuda_cflags=NVCC_FLAGS,
        extra_cflags=["-O3"],
        build_directory=str(d),
        verbose=True,
    )


print("Compiling src_1-7_simplified for SM120 (Blackwell)...", flush=True)
ext = compile_k("k17s_sm120", _KDIR / "src_1-7_simplified")
print("Done.\n")

# --------------------------------------------------------------------------- #
# Kernel progression configs (kernels 1-7 as defined in kernel_configs.py)
# --------------------------------------------------------------------------- #
progression_cfgs = get_kernel_progression_configs()
LABELS = [
    "K1: Base (async only)",
    "K2: + Swizzling",
    "K3: + Eager K/V load",
    "K4: + Interleaved LD/ST (streaming mma)",
    "K5: + Double buffering",
    "K6: + Optimized softmax (exp2)",
    "K7: Auto-tuned (best of above)",
]

# --------------------------------------------------------------------------- #
# Extra Blackwell-optimized configs with Bc=128 for higher arithmetic intensity
# --------------------------------------------------------------------------- #
# Bc=128 gives 33% better arithmetic intensity vs Bc=64:
#   Bc=64:  AI = 2*Br*d*Bc / (2*(Br+Bc)*d*2) = 16 FLOP/byte  (Br=Bc=64)
#   Bc=128: AI = 2*64*128*128 / (2*(64+128)*128*2) ≈ 21.3 FLOP/byte
# smem usage: (Br + 2*Bc)*d*2 = (64+256)*128*2 = 81,920 bytes < 100 KB ✓
BL_CFGS = [
    # (FP16, d=128, Br=64, Bc=128, NW=4): eager+swizzled+streaming(0,2,2)+opt_softmax
    FlashForwardKernelConfig(DType.FP16, 128, 64, 128, 4,
                             True, True, True, 0, 2, 2, False, True),
    # Same with double-buffer
    FlashForwardKernelConfig(DType.FP16, 128, 64, 128, 4,
                             True, True, True, 0, 2, 2, True, True),
    # (FP16, d=128, Br=64, Bc=128, NW=8): 8 warps — more latency hiding
    FlashForwardKernelConfig(DType.FP16, 128, 64, 128, 8,
                             True, True, True, 0, 2, 2, False, True),
    # (FP16, d=128, Br=128, Bc=128, NW=8): max tile, smem=98,304 B < 100 KB
    FlashForwardKernelConfig(DType.FP16, 128, 128, 128, 8,
                             True, True, True, 2, 2, 2, True, True),
]

SEQ_LENS  = [512, 1024, 2048, 4096, 8192]
N_HEADS, D_HEAD = 16, 128
BATCH = {512: 16, 1024: 16, 2048: 16, 4096: 8, 8192: 4}
NW, NR = 5, 20

_flush = torch.empty(int(200 * 1024**2), dtype=torch.int8, device="cuda")  # 200 MB


def bench(fn):
    for _ in range(NW):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(NR):
        _flush.zero_()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.mean(ts)


@torch.inference_mode()
def run():
    # ------------------------------------------------------------------ #
    # Section 1: Progression kernels 1-7
    # ------------------------------------------------------------------ #
    print("=" * 80)
    print("SECTION 1 — Kernel Progression 1-7 (SM120 native)")
    print("=" * 80)
    header = f"{'Config':<48}" + "  ".join(f"seq={s:>5}" for s in SEQ_LENS)
    print(header)
    print("-" * len(header))

    for lbl, cfg in zip(LABELS, progression_cfgs):
        row = {}
        for seq_len in SEQ_LENS:
            B = BATCH[seq_len]
            q = torch.randn(B, seq_len, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
            k, v = torch.randn_like(q), torch.randn_like(q)
            o = torch.empty_like(q)
            ref_ms = bench(lambda: F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)))
            try:
                our_ms = bench(lambda: ext.forward(cfg, q, k, v, o, False))
                row[seq_len] = 100 * ref_ms / our_ms
            except Exception as e:
                row[seq_len] = float("nan")
                print(f"  [WARN] {lbl} seq={seq_len}: {e}")
        vals = "  ".join(f"{row[s]:6.1f}%" for s in SEQ_LENS)
        print(f"{lbl:<48}{vals}")
    print()

    # ------------------------------------------------------------------ #
    # Section 2: Blackwell Bc=128 optimized configs
    # ------------------------------------------------------------------ #
    print("=" * 80)
    print("SECTION 2 — Blackwell-optimized Bc=128 tile configs (SM120)")
    print("=" * 80)
    BL_LABELS = [
        "BL1: Bc=128 NW=4 stream(0,2,2)+osfx",
        "BL2: Bc=128 NW=4 stream(0,2,2)+dbuf+osfx",
        "BL3: Bc=128 NW=8 stream(0,2,2)+osfx",
        "BL4: Br=Bc=128 NW=8 stream(2,2,2)+dbuf+osfx",
    ]
    print(header)
    print("-" * len(header))

    for lbl, cfg in zip(BL_LABELS, BL_CFGS):
        row = {}
        for seq_len in SEQ_LENS:
            if seq_len < cfg.B_r or seq_len < cfg.B_c:
                row[seq_len] = float("nan")
                continue
            B = BATCH[seq_len]
            q = torch.randn(B, seq_len, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
            k, v = torch.randn_like(q), torch.randn_like(q)
            o = torch.empty_like(q)
            ref_ms = bench(lambda: F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)))
            try:
                our_ms = bench(lambda: ext.forward(cfg, q, k, v, o, False))
                row[seq_len] = 100 * ref_ms / our_ms
            except Exception as e:
                row[seq_len] = float("nan")
                print(f"  [WARN] {lbl} seq={seq_len}: {e}")
        vals = "  ".join(f"{row[s]:6.1f}%" if not (row[s] != row[s]) else "   n/a " for s in SEQ_LENS)
        print(f"{lbl:<48}{vals}")
    print()

    # ------------------------------------------------------------------ #
    # Correctness check
    # ------------------------------------------------------------------ #
    print("=" * 80)
    print("CORRECTNESS CHECK (K7, BL1, BL4 vs SDPA reference)")
    print("=" * 80)
    check_cfgs = [
        ("K7 (auto-tuned)",         progression_cfgs[-1]),
        ("BL1 Bc=128 NW=4 stream",  BL_CFGS[0]),
        ("BL4 Br=Bc=128 NW=8",      BL_CFGS[3]),
    ]
    for name, cfg in check_cfgs:
        B, S = 4, 1024
        q = torch.randn(B, S, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
        k, v, o = torch.randn_like(q), torch.randn_like(q), torch.empty_like(q)
        ref = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        try:
            ext.forward(cfg, q, k, v, o, False)
            err = (o - ref).abs().max().item()
            status = "OK" if err < 0.05 else "FAIL"
            print(f"  {name:<38} max|out-ref| = {err:.4e}  [{status}]")
        except Exception as e:
            print(f"  {name:<38} ERROR: {e}")


run()
