# diffusion_attn Benchmarks

## INT8 QK SageAttention — SM86

| Kernel | head_dim | dtype (QK / V) | GPU | TOPS | vs BF16 baseline |
|--------|----------|----------------|-----|------|------------------|
| INT8-QK SageAttn | 64 | INT8 / FP16 | RTX 3060 (sm86) | — | — |
| INT8-QK SageAttn | 128 | INT8 / FP16 | RTX 3060 (sm86) | — | — |

*Numbers being populated. Run `experiments/bench_sageattn.py` to reproduce.*

---

## Why INT8 QK quantization matters for video DiT inference

Video diffusion transformers (HunyuanVideo, CogVideoX, LTX-Video) operate on 3D spatio-temporal token grids with shape **T × H × W** (time × height × width). This produces attention patterns that are fundamentally different from LLM inference:

**Non-causal, full-grid attention.** Unlike autoregressive LLMs, video DiTs attend over every spatial and temporal position simultaneously. There is no causal mask — the full T×H×W token grid is materialized for each attention layer. This means QK GEMM dominates memory bandwidth, not the KV cache read that LLMs worry about.

**3D spatial-temporal redundancy.** Adjacent frames share high spatial correlation. This means the Q and K activation distributions tend to be smoother and more Gaussian-like across the temporal axis, making INT8 per-warp quantization numerically stable without the outlier issues common in LLM key/query activations.

**Non-causal sparsity structure.** LLM quantized attention research typically focuses on KV cache compression under causal masking. For video DiTs the bottleneck is the full-attention QK matmul over large spatial grids (e.g., a 720p 5-second video at 24fps produces tens of thousands of tokens per layer). INT8 quantizing Q and K directly reduces this 2× in bandwidth with minimal accuracy loss, while keeping V in FP16 preserves output fidelity.

**Head dimensions.** Video DiT models use head_dim=64 and head_dim=128, the same as transformer attention but without GQA — every head sees the full token grid. Benchmarking both head dims captures the realistic operating range for this workload.
