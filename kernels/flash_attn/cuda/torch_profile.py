#!/usr/bin/env python3
"""
torch_profile.py — torch.profiler timing for K1 vs K16 flash attention.

Run (WSL):
  cd /mnt/d/GITHUB/Mini-Attention
  LD_LIBRARY_PATH=/root/fa_env/lib/python3.12/site-packages/torch/lib \
    /root/fa_env/bin/python kernels/flash_attn/cuda/torch_profile.py
"""

import os, sys
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity

_HERE  = Path(__file__).resolve().parent
_KDIR  = _HERE / "kernels"
_BUILD = _HERE / "build"

sys.path.insert(0, str(_HERE / "py"))
from flash_helpers.kernel_configs import get_kernel_progression_configs

NVCC = [
    "-std=c++20", "--use_fast_math", "--expt-relaxed-constexpr",
    "-gencode", "arch=compute_86,code=sm_86", "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

B, H, N, D = 1, 4, 1024, 128


def _compile(tag, src_dir):
    bd = _BUILD / tag
    bd.mkdir(parents=True, exist_ok=True)
    return load(name=tag, sources=[str(src_dir / "flash_attention.cu")],
                extra_include_paths=[str(src_dir / "include")],
                extra_cuda_cflags=NVCC, extra_cflags=["-O3"],
                build_directory=str(bd), verbose=False)


def run_profile(label, fn, q, k, v):
    for _ in range(5): fn(q, k, v)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with record_function(label):
            for _ in range(20):
                fn(q, k, v)
    torch.cuda.synchronize()
    return prof


def fmt_table(prof, label):
    print(f"\n=== {label} ===")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=8,
        max_name_column_width=40,
    ))


def main():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  SM{props.major}{props.minor}")
    print(f"Config: B={B} H={H} N={N} D={D}  (FP16)\n")

    print("Compiling...", flush=True)
    ext17 = _compile("k17p", _KDIR / "src_1-7")
    ext16 = _compile("k16p", _KDIR / "src_16")
    prog    = get_kernel_progression_configs()
    cfg_k1  = prog[0]   # K1: base
    cfg_k16 = prog[6]   # K7 auto-tune config -- supported by all kernels incl. K16
    print("  OK\n")

    q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    o1  = torch.empty_like(q)
    o16 = torch.empty_like(q)

    def run_k1(q, k, v):  ext17.forward(cfg_k1,  q, k, v, o1,  False)
    def run_k16(q, k, v): ext16.forward(cfg_k16, q, k, v, o16, False)
    def run_sdpa(q, k, v):
        F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2))

    for label, fn in [
        ("K1 (Base Implementation)", run_k1),
        ("K16 (Static GMEM Stride)",  run_k16),
        ("SDPA (PyTorch reference)",  run_sdpa),
    ]:
        prof = run_profile(label, fn, q, k, v)
        fmt_table(prof, label)


if __name__ == "__main__":
    main()
