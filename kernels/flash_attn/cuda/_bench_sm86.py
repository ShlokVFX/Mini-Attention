#!/usr/bin/env python3
"""
Benchmark: kernels/sm_86 — original src_1-7_simplified targeting RTX 3060 (SM86).

Usage:
    TORCH_CUDA_ARCH_LIST=8.6 python kernels/flash_attn/cuda/_bench_sm86.py
"""
import os, statistics, sys
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_KDIR = _HERE / "kernels"
sys.path.insert(0, str(_HERE / "py"))
from flash_helpers.kernel_configs import get_kernel_progression_configs

NVCC_FLAGS = [
    "-std=c++20", "--use_fast_math", "--expt-relaxed-constexpr",
    "-gencode", "arch=compute_86,code=sm_86",
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

def compile_k(tag, src_dir):
    d = _HERE / "build" / tag
    d.mkdir(parents=True, exist_ok=True)
    return load(name=tag, sources=[str(src_dir / "flash_attention.cu")],
                extra_include_paths=[str(src_dir / "include")],
                extra_cuda_cflags=NVCC_FLAGS, extra_cflags=["-O3"],
                build_directory=str(d), verbose=False)

print("Compiling sm_86 (original src_1-7_simplified)...", flush=True)
ext = compile_k("sm86", _KDIR / "sm_86")
print("Done.\n")

cfgs = get_kernel_progression_configs()
LABELS = [
    "K1: Base (async only)",
    "K2: + Swizzling",
    "K3: + Eager K/V load",
    "K4: + Interleaved LD/ST",
    "K5: + Double buffering",
    "K6: + Optimized softmax",
    "K7: Auto-tuned",
]
SEQ_LENS = [512, 1024, 2048, 4096, 8192]
N_HEADS, D_HEAD = 16, 128
BATCH = {512: 16, 1024: 16, 2048: 16, 4096: 8, 8192: 4}
NW, NR = 5, 20

_flush = torch.empty(int(50 * 1024**2), dtype=torch.int8, device="cuda")

def bench(fn):
    for _ in range(NW): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(NR):
        _flush.zero_(); torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.mean(ts)

@torch.inference_mode()
def run():
    col = "  ".join(f"seq={s:>5}" for s in SEQ_LENS)
    print(f"{'Config':<42}  {col}")
    print("-" * (44 + len(col)))
    for lbl, cfg in zip(LABELS, cfgs):
        row = {}
        for seq_len in SEQ_LENS:
            B = BATCH[seq_len]
            q = torch.randn(B, seq_len, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
            k, v, o = torch.randn_like(q), torch.randn_like(q), torch.empty_like(q)
            ref_ms = bench(lambda: F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)))
            our_ms = bench(lambda: ext.forward(cfg, q, k, v, o, False))
            row[seq_len] = 100 * ref_ms / our_ms
        vals = "  ".join(f"{row[s]:6.1f}%" for s in SEQ_LENS)
        print(f"{lbl:<42}  {vals}")

    print("\n--- Correctness (K7, seq=1024) ---")
    cfg7 = cfgs[-1]
    q = torch.randn(8, 1024, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
    k, v, o = torch.randn_like(q), torch.randn_like(q), torch.empty_like(q)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    ext.forward(cfg7, q, k, v, o, False)
    err = (o - ref).abs().max().item()
    print(f"  max|out-ref| = {err:.4e}  {'OK' if err < 0.05 else 'FAIL'}")

run()
