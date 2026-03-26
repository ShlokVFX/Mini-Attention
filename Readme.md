# Mini-Attention

## Updates

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
- StreamingLLM — Xiao et al. 2023 · https://arxiv.org/abs/2309.17453
- PagedAttention / vLLM — Kwon et al. 2023 · https://arxiv.org/abs/2309.06180
- GQA — Ainslie et al. 2023 · https://arxiv.org/abs/2305.13245
- kTransformer — Thu-KEG 2024 · https://github.com/kvcache-ai/ktransformers
- MXFP4 / Blackwell — NVIDIA 2024 · https://arxiv.org/abs/2408.11068
