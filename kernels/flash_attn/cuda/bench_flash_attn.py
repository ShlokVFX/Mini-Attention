"""Benchmark v1 vs v2 flash attention CUDA kernels, save figures to figures/."""
import time, sys, os
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- load both kernels ---
import importlib.util

def _load(path):
    spec = importlib.util.spec_from_file_location("_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dir = os.path.dirname(os.path.abspath(__file__))
v1 = _load(os.path.join(_dir, "fp32_flash_attn_sm86.py"))
v2 = _load(os.path.join(_dir, "fp32_flash_attn_sm86_wmma.py"))
flash_v1 = v1.flash_attn
flash_v2  = v2.flash_attn_v2
reference = v1.reference

FIGURES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "figures")
os.makedirs(FIGURES, exist_ok=True)

def bench(fn, q, k, v, warmup=20, iters=200):
    for _ in range(warmup): fn(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn(q, k, v)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms

B, H, D = 2, 8, 64
seq_lens = [128, 256, 512, 1024, 2048]

rows = []
print(f"{'N':>6}  {'v1 (ms)':>10}  {'v2 (ms)':>10}  {'speedup':>8}  {'max|err|':>10}")
for N in seq_lens:
    q = torch.randn(B, H, N, D, device='cuda')
    k = torch.randn(B, H, N, D, device='cuda')
    v = torch.randn(B, H, N, D, device='cuda')
    t1 = bench(flash_v1, q, k, v)
    t2 = bench(flash_v2, q, k, v)
    err = (flash_v2(q, k, v) - flash_v1(q, k, v)).abs().max().item()
    rows.append(dict(N=N, v1_ms=t1, v2_ms=t2, speedup=t1/t2, max_err=err))
    print(f"{N:>6}  {t1:>10.3f}  {t2:>10.3f}  {t1/t2:>7.2f}x  {err:>10.6f}")

df = pd.DataFrame(rows)

# --- latency vs N ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(df.N, df.v1_ms, marker='o', label='v1 (scalar, fp32 shmem, BLOCK_Q=32)')
ax.plot(df.N, df.v2_ms, marker='s', label='v2 (float4, fp16 shmem, BLOCK_Q=64)')
ax.set_xlabel('Sequence length N'); ax.set_ylabel('Latency (ms)')
ax.set_title(f'Flash Attention CUDA  B={B} H={H} D={D}  RTX 3060 (SM86)')
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, 'flash_attn_latency_vs_N.png'), dpi=150)
print(f"\nSaved latency figure → figures/flash_attn_latency_vs_N.png")

# --- speedup bar ---
fig2, ax2 = plt.subplots(figsize=(6, 3.5))
ax2.bar(df.N.astype(str), df.speedup, color='steelblue')
ax2.axhline(1.0, color='black', linestyle='--', linewidth=0.8)
ax2.set_xlabel('Sequence length N'); ax2.set_ylabel('Speedup v2/v1')
ax2.set_title('v2 speedup over v1  (>1 = faster)')
fig2.tight_layout()
fig2.savefig(os.path.join(FIGURES, 'flash_attn_speedup.png'), dpi=150)
print(f"Saved speedup figure  → figures/flash_attn_speedup.png")
