#!/usr/bin/env python3
"""
bench_flash_attn.py -- latency table + figures for FP16 flash attention iterations.

Compares K1 (base), K7 (auto-tune), K15 (peak) vs PyTorch SDPA.
Generates:
  figures/flash_attn_latency_vs_N.png
  figures/flash_attn_speedup.png

Run (WSL):
  cd /mnt/d/GITHUB/Mini-Attention
  LD_LIBRARY_PATH=/root/fa_env/lib/python3.12/site-packages/torch/lib \
    /root/fa_env/bin/python kernels/flash_attn/cuda/bench_flash_attn.py
"""

import os, sys, statistics
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_HERE  = Path(__file__).resolve().parent
_KDIR  = _HERE / "kernels"
_BUILD = _HERE / "build"
_FIGS  = Path(__file__).resolve().parents[3] / "figures"
_FIGS.mkdir(exist_ok=True)

sys.path.insert(0, str(_HERE / "py"))
from flash_helpers.kernel_configs import get_kernel_progression_configs, get_kernel_configs, DType

NVCC = [
    "-std=c++20", "--use_fast_math", "--expt-relaxed-constexpr",
    "-gencode", "arch=compute_86,code=sm_86", "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

SEQ_LENS  = [128, 256, 512, 1024, 2048, 4096]
B, H, D   = 4, 16, 128
N_WARMUP  = 10
N_REPEAT  = 100

_l2 = torch.empty(int(50 * 1024**2), dtype=torch.int8, device="cuda")

def flush():  _l2.zero_()

def _compile(tag, src_dir):
    bd = _BUILD / tag
    bd.mkdir(parents=True, exist_ok=True)
    return load(name=tag, sources=[str(src_dir / "flash_attention.cu")],
                extra_include_paths=[str(src_dir / "include")],
                extra_cuda_cflags=NVCC, extra_cflags=["-O3"],
                build_directory=str(bd), verbose=False)

def _best_cfg(ext):
    cfgs = [c for c in get_kernel_configs("all")
            if c.dtype == DType.FP16 and c.d_head == D][:40]
    q = torch.randn(B, 1024, H, D, dtype=torch.float16, device="cuda")
    k, v, o = torch.randn_like(q), torch.randn_like(q), torch.empty_like(q)
    best, bms = None, float("inf")
    for c in cfgs:
        try:
            for _ in range(3): ext.forward(c, q, k, v, o, False)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(3): ext.forward(c, q, k, v, o, False)
            e1.record(); torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / 3
            if ms < bms: bms, best = ms, c
        except Exception: pass
    return best

@torch.inference_mode()
def bench(fn, seq_len):
    q = torch.randn(B, seq_len, H, D, dtype=torch.float16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    for _ in range(N_WARMUP): fn(q, k, v)
    torch.cuda.synchronize()
    times = []
    for _ in range(N_REPEAT):
        flush()
        torch.cuda._sleep(200_000)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); fn(q, k, v); e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
    return statistics.median(times)

def sdpa(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    ).transpose(1,2)

def make_runner(ext, cfg):
    o = None
    def run(q, k, v):
        nonlocal o
        if o is None or o.shape != q.shape:
            o = torch.empty_like(q)
        ext.forward(cfg, q, k, v, o, False)
    return run

def main():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  SM{props.major}{props.minor}  {props.multi_processor_count} SMs\n")

    print("Compiling kernels...", flush=True)
    ext17 = _compile("k17b", _KDIR / "src_1-7")
    ext15 = _compile("k15b", _KDIR / "src_15")
    prog  = get_kernel_progression_configs()
    cfg_k1  = prog[0]   # K1: base
    cfg_k7  = prog[6]   # K7: auto-tune
    cfg_k15 = _best_cfg(ext15)
    print("  OK\n")

    runners = {
        "SDPA (ref)": sdpa,
        "K1  Base":   make_runner(ext17, cfg_k1),
        "K7  Auto":   make_runner(ext17, cfg_k7),
        "K15 Peak":   make_runner(ext15, cfg_k15),
    }

    results = {name: {} for name in runners}

    col_w = 14
    header = f"{'N':>6}  " + "".join(f"{n:>{col_w}}" for n in runners)
    print(header)
    print("-" * len(header))

    for N in SEQ_LENS:
        row = f"{N:>6}  "
        for name, fn in runners.items():
            ms = bench(fn, N)
            results[name][N] = ms
            row += f"{ms:>{col_w-3}.3f} ms "
        print(row)

    print()
    print(f"{'N':>6}  " + "".join(f"{'vs SDPA':>{col_w}}" for _ in list(runners)[1:]))
    print("-" * (6 + 2 + col_w * 3))
    for N in SEQ_LENS:
        ref = results["SDPA (ref)"][N]
        row = f"{N:>6}  "
        for name in list(runners)[1:]:
            rel = ref / results[name][N] * 100
            row += f"{rel:>{col_w-3}.1f} %   "
        print(row)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"SDPA (ref)": "#888888", "K1  Base": "#e74c3c",
              "K7  Auto":   "#f39c12", "K15 Peak": "#2ecc71"}
    lss    = {"SDPA (ref)": "--",      "K1  Base": "-.",
              "K7  Auto":   ":",       "K15 Peak": "-"}

    # Fig 1: latency vs N
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, data in results.items():
        xs = list(data.keys())
        ys = list(data.values())
        ax.plot(xs, ys, marker="o", label=name,
                color=colors[name], linestyle=lss[name], linewidth=2)
    ax.set_xlabel("Sequence length N", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(f"FP16 Flash Attention -- RTX 3060 (SM86)\nB={B} H={H} D={D}", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(SEQ_LENS)
    ax.set_xticklabels([str(n) for n in SEQ_LENS])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = _FIGS / "flash_attn_latency_vs_N.png"
    fig.savefig(out1, dpi=150)
    print(f"\nSaved: {out1}")
    plt.close()

    # Fig 2: speedup bar chart (harmonic mean across all N)
    def hmean(vals): return len(vals) / sum(1/v for v in vals)
    ref_hm   = hmean(list(results["SDPA (ref)"].values()))
    names_k  = [n for n in runners if n != "SDPA (ref)"]
    speedups = [ref_hm / hmean(list(results[n].values())) * 100 for n in names_k]
    short    = ["K1\nBase", "K7\nAuto-Tune", "K15\nPeak"]
    clrs     = ["#e74c3c", "#f39c12", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(short, speedups, color=clrs, edgecolor="black", linewidth=0.8, width=0.5)
    ax.axhline(100, color="gray", linestyle="--", linewidth=1.5, label="SDPA = 100%")
    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{sp:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Performance vs PyTorch SDPA (%)", fontsize=12)
    ax.set_title(
        "FP16 Flash Attention Speedup -- RTX 3060 (SM86)\n(harmonic mean, N=128..4096)",
        fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(speedups) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = _FIGS / "flash_attn_speedup.png"
    fig.savefig(out2, dpi=150)
    print(f"Saved: {out2}")
    plt.close()


if __name__ == "__main__":
    main()
