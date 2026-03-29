# Mini-Attention

**FlashAttention-2/3 reimplemented from scratch in CUDA and Triton across sm86 (A10/RTX3090), sm90 (H100), and sm100 (Blackwell, WIP). Includes INT8/FP8 quantized attention kernels for video diffusion transformers.**

## Kernels

| Folder | Description |
|--------|-------------|
| `kernels/flash_attn` | Flash Attention 2/3 implemented from scratch: K1–K17 optimization progression in CUDA + Triton port, targeting sm86 and sm90 |
| `kernels/diffusion_attn` | INT8 QK quantized attention (SageAttention) for video DiT inference; CUDA + Triton + PyTorch reference |
| `kernels/standard` | MHA, GQA, MQA variants — CUDA, Triton, and PyTorch implementations for correctness baselines |

## Benchmarks

| Kernel | Arch | Dtype | TFLOPS | % Peak MFU | Notes |
|--------|------|-------|--------|------------|-------|
| Flash Attention K1 (base) | sm86 | FP16 | — | — | baseline async |
| Flash Attention K7 (auto-tuned) | sm86 | FP16 | — | — | double-buffered, bank-conflict-free |
| Flash Attention | sm90 | FP16 | — | — | placeholder — hardware pending |
| SageAttention INT8 QK | sm86 | INT8/FP16 | — | — | head_dim=64, video DiT workload |
| SageAttention INT8 QK | sm86 | INT8/FP16 | — | — | head_dim=128, video DiT workload |

*Benchmark numbers being populated — see `experiments/` for `_bench_sm86.py`, `bench_sageattn.py`, and NCU scripts.*

## Profiling

NSight Systems timeline and roofline charts are being added. Methodology:

- `experiments/torch_profile.py` — CPU-side kernel timeline
- `kernels/flash_attn/cuda/sm86/ncu_metrics.py` — NCU analytical metrics (achieved occupancy, DRAM BW, compute BW)
- `kernels/flash_attn/cuda/sm86/_pr_ncu.py` — per-kernel NCU report

*Charts being added — figures/ directory is populated as benchmarks are finalized.*

## WIP

- **sm100 / B200 Blackwell branch** — in progress, pending hardware access. Kernels are staged and will be benchmarked on B200 when available.
- **sm120 / RTX 5090 Blackwell** — initial L2::256B + fast reciprocal kernel landed in [PR #10](https://github.com/ShlokVFX/Mini-Attention/pull/10); further tuning in progress.

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
- MXFP4 / Blackwell — NVIDIA 2024 · https://arxiv.org/abs/2408.11068
