#!/usr/bin/env python3
"""
Benchmark: kernels/b200 — SM_100-optimized FlashAttention for NVIDIA B200.

Architecture notes:
  - B200 (GB200 Blackwell datacenter): SM_100, 228 KB smem/SM, ~8 TB/s HBM3e
  - 256-byte L2 sectors (same as SM_120 consumer Blackwell)
  - WGMMA available (warpgroup-level MMA, inherited from Hopper SM_90)
  - Larger tile sizes (Bc=256) enabled by 228 KB smem

Compile target:  -gencode arch=compute_100,code=sm_100
Baseline:        Best available SDPA (cuDNN → FlashAttention → Math fallback)

Usage:
    source /workspace/Mini-Attention/.venv/bin/activate
    python kernels/flash_attn/cuda/b200/_bench_b200.py
"""
import os, statistics, sys
from pathlib import Path

# Override arch list for SM100 (B200 datacenter Blackwell)
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load

_HERE  = Path(__file__).resolve().parent   # .../cuda/b200/
_KDIR  = _HERE / "kernel"                  # .../cuda/b200/kernel/
sys.path.insert(0, str(_HERE / "py"))
from flash_helpers.kernel_configs import (
    get_kernel_progression_configs,
    get_b200_kernel_configs,
    get_wgmma_kernel_configs,
    FlashForwardKernelConfig,
    DType,
    calc_total_flop,
)

# --------------------------------------------------------------------------- #
# NVCC flags: target SM_100 (B200 datacenter Blackwell) natively.
#   • 228 KB shared memory per SM
#   • 256-byte L2 cache sectors
#   • WGMMA instructions (inherited from SM_90 Hopper)
#   • HBM3e: ~8 TB/s memory bandwidth
# --------------------------------------------------------------------------- #
NVCC_FLAGS = [
    "-std=c++20",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    # Native SM_100 binary code (B200 datacenter Blackwell)
    "-gencode", "arch=compute_100,code=sm_100",
    # SM_90 PTX as forward-compatible fallback
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


print("Compiling b200 (SM_100 datacenter Blackwell-optimized)...", flush=True)
ext = compile_k("b200", _KDIR)
print("Done.\n")

# --------------------------------------------------------------------------- #
# Kernel progression labels (K1–K7)
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
# B200-optimized configs: BL1–BL4 (Bc=128) and B2_1–B2_4 (Bc=256)
# --------------------------------------------------------------------------- #
b200_cfgs = get_b200_kernel_configs()
wgmma_cfgs = get_wgmma_kernel_configs()

B200_LABELS = [
    "BL1: Bc=128 NW=4 stream(0,2,2)+osfx",
    "BL2: Bc=128 NW=4 stream(0,2,2)+dbuf+osfx",
    "BL3: Bc=128 NW=8 stream(0,2,2)+osfx",
    "BL4: Br=Bc=128 NW=8 stream(2,2,2)+dbuf+osfx",
    "B2_1: Bc=256 NW=8 stream(0,0,4)+osfx       [B200 EXCL]",
    "B2_2: Bc=256 NW=8 stream(0,2,4)+osfx       [B200 EXCL]",
    "B2_3: Br=128,Bc=256 NW=8 stream(0,2,2)+osfx[B200 EXCL]",
    "B2_4: Br=Bc=256 NW=8 stream(2,2,2)+dbuf    [B200 EXCL]",
]

SEQ_LENS  = [512, 1024, 2048, 4096, 8192]
N_HEADS, D_HEAD = 16, 128
BATCH = {512: 16, 1024: 16, 2048: 16, 4096: 8, 8192: 4}
NW, NR = 5, 20

# Large cache flush (256 MB) to prevent L2 / HBM residency bias
_flush = torch.empty(int(256 * 1024**2), dtype=torch.int8, device="cuda")


def _probe_best_backend():
    """Return fastest available SDPA backend: cuDNN > Flash > Math."""
    probe = torch.randn(1, 1, 8, 64, dtype=torch.float16, device="cuda")
    for backend in [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]:
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(probe, probe, probe)
            return backend
        except (RuntimeError, AssertionError):
            continue
    return SDPBackend.MATH


_BEST_BACKEND = _probe_best_backend()
print(f"Reference backend: {_BEST_BACKEND.name}  "
      f"{'(cuDNN — optimal B200 baseline)' if _BEST_BACKEND == SDPBackend.CUDNN_ATTENTION else '(cuDNN unavailable — using fallback)'}\n")


def ref_sdpa(q, k, v):
    with sdpa_kernel(_BEST_BACKEND):
        return F.scaled_dot_product_attention(q, k, v)


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


def tflops(seq_len, batch, n_heads, d_head, b_r, b_c, ms):
    total = calc_total_flop(batch, n_heads, seq_len, b_r, b_c, d_head)
    return total / (ms * 1e-3) / 1e12


@torch.inference_mode()
def run():
    # ------------------------------------------------------------------ #
    # Section 1: Progression K1-K7 (SM_100 native compilation)
    # ------------------------------------------------------------------ #
    print("=" * 88)
    print("SECTION 1 — Kernel Progression K1-7 (SM_100 native)")
    print(f"Baseline: {_BEST_BACKEND.name}")
    print("=" * 88)
    header = f"{'Config':<52}" + "  ".join(f"seq={s:>5}" for s in SEQ_LENS)
    print(header)
    print("-" * len(header))

    for lbl, cfg in zip(LABELS, progression_cfgs):
        row = {}
        for seq_len in SEQ_LENS:
            B = BATCH[seq_len]
            q = torch.randn(B, seq_len, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
            k, v = torch.randn_like(q), torch.randn_like(q)
            o = torch.empty_like(q)
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ref_ms = bench(lambda: ref_sdpa(qt, kt, vt))
            try:
                our_ms = bench(lambda: ext.forward(cfg, q, k, v, o, False))
                row[seq_len] = 100 * ref_ms / our_ms
            except Exception as e:
                row[seq_len] = float("nan")
                print(f"  [WARN] {lbl} seq={seq_len}: {e}")
        vals = "  ".join(f"{row[s]:6.1f}%" for s in SEQ_LENS)
        print(f"{lbl:<52}{vals}")
    print()

    # ------------------------------------------------------------------ #
    # Section 2: B200 optimized configs (Bc=128 + Bc=256)
    # ------------------------------------------------------------------ #
    print("=" * 88)
    print("SECTION 2 — B200-optimized tile configs (Bc=128 and Bc=256)")
    print(f"Baseline: {_BEST_BACKEND.name}")
    print("=" * 88)
    print(header)
    print("-" * len(header))

    for lbl, cfg in zip(B200_LABELS, b200_cfgs):
        row = {}
        for seq_len in SEQ_LENS:
            if seq_len < cfg.B_r or seq_len < cfg.B_c:
                row[seq_len] = float("nan")
                continue
            B = BATCH[seq_len]
            q = torch.randn(B, seq_len, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
            k, v = torch.randn_like(q), torch.randn_like(q)
            o = torch.empty_like(q)
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ref_ms = bench(lambda: ref_sdpa(qt, kt, vt))
            try:
                our_ms = bench(lambda: ext.forward(cfg, q, k, v, o, False))
                row[seq_len] = 100 * ref_ms / our_ms
            except Exception as e:
                row[seq_len] = float("nan")
                print(f"  [WARN] {lbl} seq={seq_len}: {e}")
        vals = "  ".join(
            f"{row[s]:6.1f}%" if row[s] == row[s] else "   n/a " for s in SEQ_LENS
        )
        print(f"{lbl:<52}{vals}")
    print()

    # ------------------------------------------------------------------ #
    # Section 3: WGMMA kernel variants (QK GEMM via wgmma.mma_async)
    # ------------------------------------------------------------------ #
    print("=" * 88)
    print("SECTION 3 — WGMMA kernels (wgmma.mma_async QK, mma.sync PV)")
    print(f"Baseline: {_BEST_BACKEND.name}")
    print("=" * 88)
    WGMMA_LABELS = [
        "KW_1: Br=64,  Bc=64  NW=4  wgmma.m64n64k16",
        "KW_2: Br=64,  Bc=128 NW=4  wgmma.m64n128k16",
        "KW_3: Br=128, Bc=64  NW=8  wgmma.m64n64k16  x2WG",
        "KW_4: Br=128, Bc=128 NW=8  wgmma.m64n128k16 x2WG",
    ]
    print(header)
    print("-" * len(header))

    for lbl, cfg in zip(WGMMA_LABELS, wgmma_cfgs):
        row = {}
        for seq_len in SEQ_LENS:
            if seq_len < cfg.B_r or seq_len < cfg.B_c:
                row[seq_len] = float("nan")
                continue
            B = BATCH[seq_len]
            q = torch.randn(B, seq_len, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
            k, v = torch.randn_like(q), torch.randn_like(q)
            o = torch.empty_like(q)
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ref_ms = bench(lambda: ref_sdpa(qt, kt, vt))
            try:
                our_ms = bench(lambda: ext.forward(cfg, q, k, v, o, False, True))
                row[seq_len] = 100 * ref_ms / our_ms
            except Exception as e:
                row[seq_len] = float("nan")
                print(f"  [WARN] {lbl} seq={seq_len}: {e}")
        vals = "  ".join(
            f"{row[s]:6.1f}%" if row[s] == row[s] else "   n/a " for s in SEQ_LENS
        )
        print(f"{lbl:<52}{vals}")
    print()

    # ------------------------------------------------------------------ #
    # Section 4: TFLOPs throughput comparison
    # ------------------------------------------------------------------ #
    print("=" * 88)
    print("SECTION 4 — Peak TFLOPs throughput (B200 best configs + WGMMA)")
    print("=" * 88)
    tflops_header = f"{'Config':<52}" + "  ".join(f"seq={s:>4}TF" for s in SEQ_LENS)
    print(tflops_header)
    print("-" * len(tflops_header))

    best_cfgs = [
        ("cuDNN SDPA (reference)",              None,           False),
        ("B2_4: Br=Bc=128 NW=8 dbuf [mma.s]",  b200_cfgs[3],   False),
        ("KW_2: Br=64,Bc=128 NW=4 [wgmma QK]", wgmma_cfgs[1],  True),
        ("KW_4: Br=128,Bc=128 NW=8 [wgmma QK]",wgmma_cfgs[3],  True),
    ]
    for lbl, cfg, use_wg in best_cfgs:
        row = {}
        for seq_len in SEQ_LENS:
            if cfg is not None and (seq_len < cfg.B_r or seq_len < cfg.B_c):
                row[seq_len] = float("nan")
                continue
            B = BATCH[seq_len]
            q = torch.randn(B, seq_len, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
            k, v = torch.randn_like(q), torch.randn_like(q)
            o = torch.empty_like(q)
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            try:
                if cfg is None:
                    ms = bench(lambda: ref_sdpa(qt, kt, vt))
                    br, bc = 64, 64
                else:
                    ms = bench(lambda: ext.forward(cfg, q, k, v, o, False, use_wg))
                    br, bc = cfg.B_r, cfg.B_c
                row[seq_len] = tflops(seq_len, B, N_HEADS, D_HEAD, br, bc, ms)
            except Exception as e:
                row[seq_len] = float("nan")
        vals = "  ".join(
            f"{row[s]:7.2f}" if row[s] == row[s] else "    n/a" for s in SEQ_LENS
        )
        print(f"{lbl:<52}{vals}")
    print()

    # ------------------------------------------------------------------ #
    # Section 5: Correctness check
    # ------------------------------------------------------------------ #
    print("=" * 88)
    print("CORRECTNESS CHECK (vs cuDNN/Flash reference)")
    print("=" * 88)
    check_cfgs = [
        # (label, cfg, use_wgmma)
        ("K7 (auto-tuned)",                  progression_cfgs[-1], False),
        ("BL1 Bc=128 NW=4 stream [mma.s]",   b200_cfgs[0],         False),
        ("BL4 Br=Bc=128 NW=8 dbuf [mma.s]",  b200_cfgs[3],         False),
        ("B2_1 Bc=256 NW=8 stream(0,0,4)",   b200_cfgs[4],         False),
        ("B2_3 Br=128,Bc=256 NW=8",          b200_cfgs[6],         False),
        ("KW_1 Br=64,Bc=64  NW=4 [wgmma]",   wgmma_cfgs[0],        True),
        ("KW_2 Br=64,Bc=128 NW=4 [wgmma]",   wgmma_cfgs[1],        True),
        ("KW_3 Br=128,Bc=64  NW=8 [wgmma]",  wgmma_cfgs[2],        True),
        ("KW_4 Br=128,Bc=128 NW=8 [wgmma]",  wgmma_cfgs[3],        True),
    ]
    for name, cfg, use_wg in check_cfgs:
        B, S = 4, 1024
        if S < cfg.B_r or S < cfg.B_c:
            print(f"  {name:<44} skipped (seq_len < tile)")
            continue
        q = torch.randn(B, S, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
        k, v, o = torch.randn_like(q), torch.randn_like(q), torch.empty_like(q)
        ref = ref_sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        try:
            ext.forward(cfg, q, k, v, o, False, use_wg)
            err = (o - ref).abs().max().item()
            status = "OK" if err < 0.05 else "FAIL"
            print(f"  {name:<44} max|out-ref| = {err:.4e}  [{status}]")
        except Exception as e:
            print(f"  {name:<44} ERROR: {e}")


run()
