#!/usr/bin/env python3
"""Quick comparison: src_1-7 vs src_1-7_simplified across all 7 progression configs."""
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
    "-gencode", "arch=compute_86,code=sm_86", "-O3",
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

print("Compiling k17  (src_1-7)...", flush=True)
e17  = compile_k("k17",  _KDIR / "src_1-7")
print("Compiling k17s (src_1-7_simplified)...", flush=True)
e17s = compile_k("k17s", _KDIR / "src_1-7_simplified")
print("Done.\n")

cfgs = get_kernel_progression_configs()
LABELS = [
    "1. Base Implementation",
    "2. Swizzling",
    "3. Eagerly Loading K & V",
    "4. Interleaving LD/ST",
    "5. Double Buffering",
    "6. Improving FP32 Throughput",
    "7. Auto-Tuning",
]
SEQ_LENS = [512, 1024, 2048, 4096, 8192]
N_HEADS, D_HEAD = 16, 128
BATCH = {512: 16, 1024: 16, 2048: 16, 4096: 16, 8192: 8}
NW, NR = 5, 20

_flush = torch.empty(int(50 * 1024**2), dtype=torch.int8, device="cuda")

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
    header = f"{'Config':<42}" + "  ".join(f"seq={s:>5}" for s in SEQ_LENS)
    print(header)
    print("-" * len(header))

    for lbl, cfg in zip(LABELS, cfgs):
        row17  = {}
        row17s = {}
        for seq_len in SEQ_LENS:
            B = BATCH[seq_len]
            q = torch.randn(B, seq_len, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            o = torch.empty_like(q)
            ref_ms = bench(lambda: F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)))
            ms17  = bench(lambda: e17.forward(cfg, q, k, v, o, False))
            ms17s = bench(lambda: e17s.forward(cfg, q, k, v, o, False))
            row17[seq_len]  = 100 * ref_ms / ms17
            row17s[seq_len] = 100 * ref_ms / ms17s

        vals17  = "  ".join(f"{row17[s]:6.1f}%" for s in SEQ_LENS)
        vals17s = "  ".join(f"{row17s[s]:6.1f}%" for s in SEQ_LENS)
        simplified_lbl = lbl + " (Simplified)"
        print(f"{lbl:<42}{vals17}")
        print(f"{simplified_lbl:<42}{vals17s}")
        print()

    # Correctness check on K7 config
    print("--- Correctness check (K7 config, seq=1024, B=8) ---")
    cfg7 = cfgs[6]
    q = torch.randn(8, 1024, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
    k, v, o = torch.randn_like(q), torch.randn_like(q), torch.empty_like(q)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    e17.forward(cfg7, q, k, v, o, False)
    err17 = (o - ref).abs().max().item()
    e17s.forward(cfg7, q, k, v, o, False)
    err17s = (o - ref).abs().max().item()
    print(f"  k17  max|out-ref| = {err17:.4e}  {'OK' if err17 < 0.05 else 'FAIL'}")
    print(f"  k17s max|out-ref| = {err17s:.4e}  {'OK' if err17s < 0.05 else 'FAIL'}")


run()
