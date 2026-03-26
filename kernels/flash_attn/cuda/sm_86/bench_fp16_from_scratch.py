#!/usr/bin/env python3
"""
FP16 Flash Attention from scratch — all 16 kernel iterations on RTX 3060 (SM86).

References kernel sources from flash_attention_from_scratch/ in this repo.
Run from WSL (avoids Windows MSVC/cudafe++ issues):

  cd /mnt/d/GITHUB/Mini-Attention
  LD_LIBRARY_PATH=/root/fa_env/lib/python3.12/site-packages/torch/lib \
    /root/fa_env/bin/python kernels/flash_attn/cuda/bench_fp16_from_scratch.py
"""

import os
import statistics
import sys
from pathlib import Path

# Pin to SM86 only — faster compilation, avoids arch-list warning
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.utils.cpp_extension import load

# Paths — all sources live alongside this file in kernels/flash_attn/cuda/
_HERE  = Path(__file__).resolve().parent
_KDIR  = _HERE / "kernels"   # kernel source dirs (src_1-7 … src_16)
_BUILD = _HERE / "build"

sys.path.insert(0, str(_HERE / "py"))
from flash_helpers.kernel_configs import (
    DType,
    FlashForwardKernelConfig,
    get_kernel_configs,
    get_kernel_progression_configs,
)

SEQ_LENS = [512, 1024, 2048, 4096, 8192]
D_HEAD   = 128
N_HEADS  = 16
BATCH_FOR_SEQ = {512: 16, 1024: 16, 2048: 16, 4096: 16, 8192: 8, 16384: 4}
N_WARMUPS = 10
N_REPEATS = 40

NVCC_FLAGS = [
    "-std=c++20",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "-gencode", "arch=compute_86,code=sm_86",
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

_l2_flush = torch.empty(int(100 * 1024 ** 2), dtype=torch.int8, device="cuda")


def flush_l2():
    _l2_flush.zero_()


def compile_kernel(tag: str, src_dir: Path):
    build_dir = _BUILD / tag
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=tag,
        sources=[str(src_dir / "flash_attention.cu")],
        extra_include_paths=[str(src_dir / "include")],
        extra_cuda_cflags=NVCC_FLAGS,
        extra_cflags=["-O3"],
        build_directory=str(build_dir),
        verbose=False,
    )


@torch.inference_mode()
def time_kernel(fn, n_warmups=N_WARMUPS, n_repeats=N_REPEATS):
    for _ in range(n_warmups):
        fn()
    torch.cuda.synchronize()

    runtimes = []
    for _ in range(n_repeats):
        flush_l2()
        torch.cuda._sleep(500_000)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        runtimes.append(start.elapsed_time(end))
    return runtimes


def sdpa_reference(q, k, v):
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    return F.scaled_dot_product_attention(qt, kt, vt).transpose(1, 2)


def bench_kernel(ext, cfg, q, k, v, o):
    return time_kernel(lambda: ext.forward(cfg, q, k, v, o, False))


def bench_sdpa(q, k, v):
    return time_kernel(lambda: sdpa_reference(q, k, v))


def harmonic_mean(values):
    if not values:
        return float("nan")
    return len(values) / sum(1.0 / v for v in values)


def calc_flops(batch, n_heads, seq_len, d_head):
    return 4 * batch * n_heads * seq_len * seq_len * d_head


# K1-K7: official progression configs from flash_helpers
_K17_LABELS = [
    "1. Base Implementation",
    "2. Swizzling",
    "3. Eagerly Loading K & V",
    "4. Interleaving LD/ST",
    "5. Double Buffering",
    "6. Improving FP32 Throughput",
    "7. Auto-Tuning",
]

# K1-K7 Simplified: same configs, simplified headers
_K17S_LABELS = [
    "1s. Base Implementation (Simplified)",
    "2s. Swizzling (Simplified)",
    "3s. Eagerly Loading K & V (Simplified)",
    "4s. Interleaving LD/ST (Simplified)",
    "5s. Double Buffering (Simplified)",
    "6s. Improving FP32 Throughput (Simplified)",
    "7s. Auto-Tuning (Simplified)",
]


def k17_configs():
    cfgs = get_kernel_progression_configs()
    return list(zip(_K17_LABELS, cfgs))


def k17s_configs():
    cfgs = get_kernel_progression_configs()
    return list(zip(_K17S_LABELS, cfgs))


def best_config_at(ext, seq_len):
    """Quick scan of all FP16 configs; return fastest."""
    all_cfgs = get_kernel_configs("all")
    fp16_cfgs = [c for c in all_cfgs if c.dtype == DType.FP16][:40]

    batch = BATCH_FOR_SEQ[seq_len]
    q = torch.randn(batch, seq_len, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    o = torch.empty_like(q)

    best_cfg, best_ms = None, float("inf")
    for cfg in fp16_cfgs:
        try:
            for _ in range(3):
                ext.forward(cfg, q, k, v, o, False)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                ext.forward(cfg, q, k, v, o, False)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / 3
            if ms < best_ms:
                best_ms = ms
                best_cfg = cfg
        except Exception:
            pass
    return best_cfg


KERNEL_DIRS = [
    ("k17",  _KDIR / "src_1-7"),
    ("k17s", _KDIR / "src_1-7_simplified"),
    ("k8",   _KDIR / "src_8"),
    ("k9",   _KDIR / "src_9"),
    ("k10",  _KDIR / "src_10"),
    ("k11",  _KDIR / "src_11"),
    ("k12",  _KDIR / "src_12"),
    ("k13",  _KDIR / "src_13"),
    ("k14",  _KDIR / "src_14"),
    ("k15",  _KDIR / "src_15"),
    ("k16",  _KDIR / "src_16"),
]

ROW_NAMES = {
    "k17":  _K17_LABELS,
    "k17s": _K17S_LABELS,
    "k8":   ["8. Reducing IADD3/LOP3/SHF"],
    "k9":  ["9. Reducing IMAD.MOV/MOV"],
    "k10": ["10. Removing CSRZ + Opt Softmax"],
    "k11": ["11. Encoded Swizzling RF→SMEM"],
    "k12": ["12. Misc Code Changes"],
    "k13": ["13. Iterating Backwards"],
    "k14": ["14. Cache Configuration"],
    "k15": ["15. Tiling along d_head"],
    "k16": ["16. Static GMEM Stride"],
}


def run_benchmark():
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}  (SM{props.major}{props.minor}, {props.multi_processor_count} SMs)")
    print(f"seq_lens={SEQ_LENS}, d_head={D_HEAD}, n_heads={N_HEADS}")
    print(f"Reference: PyTorch scaled_dot_product_attention\n")

    compiled = {}
    for tag, src_dir in KERNEL_DIRS:
        print(f"  Compiling {tag} from {src_dir.name}...", flush=True)
        try:
            compiled[tag] = compile_kernel(tag, src_dir)
            print(f"    OK {tag}")
        except Exception as e:
            print(f"    FAIL {tag}: {e}")

    print()

    results    = {}
    ref_tflops = {}

    for seq_len in SEQ_LENS:
        batch = BATCH_FOR_SEQ[seq_len]
        flops = calc_flops(batch, N_HEADS, seq_len, D_HEAD)

        q16 = torch.randn(batch, seq_len, N_HEADS, D_HEAD, dtype=torch.float16, device="cuda")
        k16 = torch.randn_like(q16)
        v16 = torch.randn_like(q16)
        o16 = torch.empty_like(q16)

        print(f"seq_len={seq_len}:")

        ref_times = bench_sdpa(q16, k16, v16)
        ref_mean  = statistics.mean(ref_times)
        ref_tflops[seq_len] = flops / (ref_mean * 1e6) / 1e3
        print(f"  Reference (SDPA): {ref_mean:.3f} ms  ({ref_tflops[seq_len]:.2f} TFLOP/s)")

        for tag, ext in compiled.items():
            if tag in ("k17", "k17s"):
                cfg_pairs = k17_configs() if tag == "k17" else k17s_configs()
                for label, cfg in cfg_pairs:
                    try:
                        times   = bench_kernel(ext, cfg, q16, k16, v16, o16)
                        mean_ms = statistics.mean(times)
                        rel     = 100 * ref_mean / mean_ms
                        print(f"  {label}: {mean_ms:.3f} ms  ({rel:.1f}%)")
                        results.setdefault(label, {})[seq_len] = rel
                    except Exception as ex:
                        print(f"  {label}: ERROR {ex}")
            else:
                label = ROW_NAMES[tag][0]
                cfg   = best_config_at(ext, seq_len)
                if cfg is None:
                    print(f"  {label}: no valid config")
                    continue
                try:
                    times   = bench_kernel(ext, cfg, q16, k16, v16, o16)
                    mean_ms = statistics.mean(times)
                    rel     = 100 * ref_mean / mean_ms
                    print(f"  {label}: {mean_ms:.3f} ms  ({rel:.1f}%)")
                    results.setdefault(label, {})[seq_len] = rel
                except Exception as ex:
                    print(f"  {label}: ERROR {ex}")
        print()

    # ── Final table ───────────────────────────────────────────────────────────
    table = PrettyTable()
    table.field_names = (
        ["Kernel Iteration"] +
        [f"seq={s}" for s in SEQ_LENS] +
        ["Harm. Mean"]
    )
    table.align["Kernel Iteration"] = "l"
    for col in table.field_names[1:]:
        table.align[col] = "r"

    for label, row_data in results.items():
        vals  = [row_data.get(s) for s in SEQ_LENS]
        valid = [v for v in vals if v is not None]
        hm    = harmonic_mean(valid) if valid else float("nan")
        table.add_row([label] + [f"{v:.1f}%" if v else "-" for v in vals] + [f"{hm:.1f}%"])

    avg_ref = statistics.mean(ref_tflops.values())
    table.add_row([""] + [""] * (len(SEQ_LENS) + 1))
    table.add_row(
        ["0. Reference (SDPA TFLOP/s)"] +
        [f"{ref_tflops.get(s, 0):.1f}" for s in SEQ_LENS] +
        [f"{avg_ref:.1f}"]
    )

    print(f"\n{'='*80}")
    print(f"RTX 3060 (SM86) — FP16 Flash Attention Iteration Performance")
    print(f"Reference: PyTorch SDPA = 100%")
    print(f"{'='*80}")
    print(table)


if __name__ == "__main__":
    run_benchmark()
