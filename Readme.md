# Mini-Attention

GPU attention kernels — RTX 3060 (SM86, 12 GB, 360 GB/s HBM)

---

## Results

### Standard Attention — tiled matmul, FP16 Tensor Cores

| | Triton `fp16_mha_tiled_sm86` | CUDA `fp32_mha_sm86` | PyTorch ref |
|---|---|---|---|
| B=2 H=8 N=512 D=64 | **0.28 ms** | 5.4 ms | 0.53 ms |
| max\|err\| | 4.9e-4 | 0.0 | — |

### Flash Attention — online softmax, O(N·D) HBM

| | Triton `fp16_flash_attn_sm86` | CUDA `fp32_flash_attn_sm86` | PyTorch ref |
|---|---|---|---|
| B=2 H=8 N=1024 D=64 | **0.19 ms** | 21.6 ms | 1.7 ms |
| peak memory | **16.9 MB** | 25.3 MB | 157–168 MB |
| memory reduction | **89%** | 84% | — |
| max\|err\| causal/non-causal | 2.0e-3 / 2.4e-4 | 1e-6 / 1e-6 | — |

### SageAttention (Diffusion) — INT8 QK, FP16 V

| | Triton `fp16_sage_attn_sm86` | CUDA `fp32_sage_attn_sm86` | PyTorch `sage_attn` |
|---|---|---|---|
| B=2 H=8 N=1024 D=64 | **0.17 ms** | 4.1 ms | 2.8 ms |
| QK bandwidth vs FP16 | 2× reduction | 2× reduction (\_\_dp4a) | 2× (simulated) |
| max\|err\| | 0.022 | 0.008 | 0.004 |

### Streaming Attention (Infinite Context) — sink=4 + window=256

| | Triton `fp16_streaming_attn_sm86` | CUDA `fp32_streaming_attn_sm86` |
|---|---|---|
| B=2 H=8 N=1024 D=64 | ✓ 2.4e-4 | ✓ 1e-6 |
| KV memory vs standard | **75% reduction** | **75% reduction** |
| KV memory (abs) | 1.1 MB | 1.1 MB vs 4.2 MB standard |

### Paged Attention (KV Cache) — page\_size=16, indirect block\_table

| | Triton `fp16_paged_attn_sm86` | CUDA `fp32_paged_attn_sm86` |
|---|---|---|
| BH=16 N=256 D=64 | ✓ 2.4e-4 | ✓ 1e-6 |
| fragmentation vs contiguous | ~4% (1 partial page/seq) | ~4% |

### Device-Aware MoE (kTransformer) — experts on CPU, gate on GPU

| | Triton `fp16_moe_sm86` | CUDA `fp32_moe_sm86` |
|---|---|---|
| 8 experts, D=256, D\_ff=512 | 9.4 MB GPU peak | 9.0 MB GPU peak |
| all-experts-on-GPU | 6.3 MB (small test) | 6.3 MB |
| LLaMA-scale (16 experts, D=4096) | ~180 MB/expert | 540 MB/expert → 10× saving |

---

## Kernel Index

```
kernels/
  standard/
    pytorch/          vanilla MHA, MQA, GQA, einsum variants
    triton/           fp16_mha_tiled_sm86.py
    cuda/             fp32_mha_sm86.py

  flash_attn/
    pytorch/          block_segmentation.py, online_softmax.py, benchmark.py
    triton/           fp16_flash_attn_sm86.py
    cuda/             fp32_flash_attn_sm86.py

  diffusion_attn/     replaces ring_attn — SageAttention-style INT8 QK
    pytorch/          sage_attn.py
    triton/           fp16_sage_attn_sm86.py
    cuda/             fp32_sage_attn_sm86.py

  Infinite_context/
    base.py           StreamingLLM PyTorch demo
    triton/           fp16_streaming_attn_sm86.py
    cuda/             fp32_streaming_attn_sm86.py

  kv_cache/
    pytorch/          paged attention from scratch, vLLM wrapper, benchmarks
    triton/           fp16_paged_attn_sm86.py
    cuda/             fp32_paged_attn_sm86.py

  kTransformer/
    base.py           device-aware MoE PyTorch demo (265 MB → 25 MB)
    triton/           fp16_moe_sm86.py
    cuda/             fp32_moe_sm86.py

  mxfp4/              mxfp4_gemm.py — FP4 GEMM targeting SM100 (Blackwell)
  ml150/              NumPy MHA reference
```

---

## Benchmarks

![attention comparison](figures/attn_compare.png)
![paged attention performance](figures/paged_attention_perf.png)
![kv cache disabled](figures/disabled_kvCache.png)
![performance comparison](figures/performance_comparison.png)

---

## Citations

Flash Attention 2 — Dao et al. 2023 · https://arxiv.org/abs/2307.08691

SageAttention — Zhang et al. 2024 · https://arxiv.org/abs/2410.02367

StreamingLLM — Xiao et al. 2023 · https://arxiv.org/abs/2309.17453

PagedAttention / vLLM — Kwon et al. 2023 · https://arxiv.org/abs/2309.06180

GQA — Ainslie et al. 2023 · https://arxiv.org/abs/2305.13245

kTransformer — Thu-KEG 2024 · https://github.com/kvcache-ai/ktransformers

MXFP4 / Blackwell — NVIDIA 2024 · https://arxiv.org/abs/2408.11068
