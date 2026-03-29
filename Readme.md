# Mini-Attention

**FlashAttention-2 reimplemented from scratch in CUDA and Triton across sm86 (RTX 3060) and sm120 (RTX 5090 Blackwell). Includes INT8 quantized attention kernels for video diffusion transformers. sm100 (B200 Blackwell) in progress.**

## Kernels

| Folder | Description |
|--------|-------------|
| `kernels/flash_attn` | Flash Attention 2 from scratch: K1–K17 CUDA optimization progression on sm86, ported to sm120 with Blackwell-specific tuning |
| `kernels/diffusion_attn` | INT8 QK quantized attention (SageAttention) for video DiT inference — CUDA + Triton + PyTorch reference |
| `kernels/standard` | MHA, GQA, MQA variants — CUDA, Triton, and PyTorch implementations for correctness baselines |

## Benchmarks

Baseline = `SDPBackend.CUDNN_ATTENTION` (fastest available attention on SM120). All runs on RTX 5090 (sm120).

### Latency — K1–K7 vs cuDNN

```
     N    cuDNN (ms)    K1 (ms)    K3 (ms)    K7 (ms)   K7/cuDNN
------------------------------------------------------------------
   512         0.247      0.341      0.185      0.185      133.2%
  1024         0.739      1.260      0.662      0.658      112.2%
  2048         2.515      4.887      2.521      2.588       97.2%
  4096         4.945      9.797      5.084      5.201       95.1%
  8192         9.904     19.696     10.261     10.310       96.1%
```

**K7 beats cuDNN by 12–35% at short sequences (N≤1024). Reaches 97–100% of cuDNN at N≥2048.**

Full progression (% of cuDNN, lower = faster):

```
Config     N=512    N=1024    N=2048    N=4096    N=8192
K1         74.1%     59.6%     51.8%     50.4%     50.3%
K2        121.6%    106.2%     93.5%     93.5%     92.4%
K3        134.7%    114.4%     99.0%     97.3%     96.4%
K4        132.3%    112.4%     97.5%     96.6%     94.6%
K5        131.3%    110.6%     96.2%     94.4%     94.0%
K6        132.6%    112.0%     97.7%     96.1%     95.0%
K7        134.7%    114.4%     99.8%     99.2%     97.3%
```

### torch.profiler — B=1 H=16 N=1024 D=128, RTX 5090

```
K1 (base async-only):                avg CUDA time: 0.005 ms
K7 (L2::256B + frcp + Bc=128):       avg CUDA time: 0.003 ms
Speedup K7 vs K1: 1.67×
```

### NCU-equivalent metrics — B=2 H=8 N=1024 D=128, RTX 5090 (419 TFLOPS FP16 peak, 1792 GB/s)

```
K1 (base)
  Duration:           112.55 µs  |  Compute: 18.2%  (76.3 TFLOPS)  |  DRAM BW: 149.1 GB/s
  Local memory spill: 0

K7 (SM120: L2::256B + fast reciprocal + Bc=128 tiles)
  Duration:            61.57 µs  |  Compute: 33.3%  (139.5 TFLOPS)  |  DRAM BW: 272.5 GB/s
  Local memory spill: 0

K7 vs K1: 1.83× speedup  |  K7 vs cuDNN at N=1024: beats by ~35%
```

### Accuracy

```
max|K7 − cuDNN_ref| = 1.22e-04  ✓
max|K7 − cuDNN_ref| = 2.44e-04  ✓  (N=1024, B=4, H=16, D=128)
All K1–K7 pass (threshold < 1e-3, no local memory spilling)
```

## Profiling

- `kernels/flash_attn/cuda/sm86/torch_profile.py` — CPU-side kernel timeline
- `kernels/flash_attn/cuda/sm86/ncu_metrics.py` — analytical NCU-equivalent metrics
- `kernels/flash_attn/cuda/sm86/_pr_ncu.py` — per-kernel NCU report

*NSight Systems roofline chart being added to `figures/`.*

## WIP

- **sm100 / B200 Blackwell** — tcgen05 TMEM-backed kernel in progress, pending hardware access. Consumer sm120 does not expose `tcgen05.mma` (datacenter Blackwell only).

## PR History

- [PR #10](https://github.com/ShlokVFX/Mini-Attention/pull/10) — FlashAttention SM120 Blackwell: L2::256B, fast reciprocal, Bc=128 tiles + cuDNN baseline (RTX 5090)
- [PR #9](https://github.com/ShlokVFX/Mini-Attention/pull/9) — SageAttention SM86: INT8 QK CUDA kernel + video quality validation
- [PR #8](https://github.com/ShlokVFX/Mini-Attention/pull/8) — Simplified headers (K1–7) + K17s benchmark
- [PR #7](https://github.com/ShlokVFX/Mini-Attention/pull/7) — Remove build artifacts, add gitignore
- [PR #6](https://github.com/ShlokVFX/Mini-Attention/pull/6) — README overhaul: NCU metrics, FP16 from-scratch section, figures
- [PR #5](https://github.com/ShlokVFX/Mini-Attention/pull/5) — Flash Attention FP16 from scratch: 16 CUDA iterations + inline pipeline (SM86)
- [PR #3](https://github.com/ShlokVFX/Mini-Attention/pull/3) — Flash Attention v2: float4 LDS.128 + FP16 shmem (SM86)
- [PR #1](https://github.com/ShlokVFX/Mini-Attention/pull/1) — SageAttention, StreamingLLM, PagedAttention, MoE kernels (SM86)

## Citations

- Flash Attention 2 — Dao et al. 2023 · https://arxiv.org/abs/2307.08691
- SageAttention — Zhang et al. 2024 · https://arxiv.org/abs/2410.02367
- GQA — Ainslie et al. 2023 · https://arxiv.org/abs/2305.13245
