import math
import torch

# SageAttention-inspired: quantize Q, K to INT8 per warp block, keep V in FP16
# INT8 QK reduces bandwidth: N × D × 1B vs N × D × 2B FP16 → 2× QK bandwidth savings
# At N=4096 (512×512 image, 8×8 patch): QK bandwidth = 4096²×1B = 16MB vs 32MB FP16
# V kept FP16: accuracy-sensitive; errors here directly affect output quality
# smooth quant: scale = max_abs / 127.0; no outlier handling needed for typical diffusion activations


def smooth_quantize(x, dim=-1):
    """Per-row INT8 quantization for Q, K.
    scale = max(|x|, dim) / 127; quantization noise ≈ scale/2 per element
    """
    scale = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-5) / 127.0
    x_int8 = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


def sage_attn_pytorch(q, k, v, causal=False):
    """SageAttention forward in PyTorch with INT8 Q, K quantization.

    Simulates the SageAttention quantization pattern in pure PyTorch.
    Real SageAttention uses CUDA INT8 IMMA (8× INT8 vs FP16 Tensor Core throughput).

    q, k, v: (B, H, N, D) fp16
    memory: QK temp = N² × 4B fp32 (dequantized scores); actual INT8 impl avoids this
    flops: same as standard; INT8 kernel saves bandwidth (1B vs 2B per QK element)
    """
    B, H, N, D = q.shape

    # quantize Q and K to INT8 per head per row
    # bandwidth to compute scores: N × D × 1B (INT8) vs N × D × 2B (FP16) → 2× saving
    q_int8, q_scale = smooth_quantize(q.float(), dim=-1)   # scale: (B, H, N, 1)
    k_int8, k_scale = smooth_quantize(k.float(), dim=-1)

    # dequantized scores: Q_int8 @ K_int8^T × (q_scale × k_scale^T) / sqrt(D)
    # INT8 → FP32 dequant: score = (sum_d q_int8[d]*k_int8[d]) × q_scale × k_scale / 127²
    scores_int = torch.matmul(q_int8.float(), k_int8.float().transpose(-2, -1))
    # dequant: dot(q_int8, k_int8) ≈ dot(q/q_scale, k/k_scale) → multiply back by scales
    scores = scores_int * (q_scale * k_scale.transpose(-2, -1)) / math.sqrt(D)

    if causal:
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))

    # softmax in FP32 for numerical stability
    probs = torch.softmax(scores, dim=-1).to(v.dtype)

    # V stays FP16: (B, H, N, N) fp16 × (B, H, N, D) fp16 → output FP16
    # FP16 V matmul memory: N² × 2B probs + N × D × 2B V = N²×2 + N×D×2 bytes
    return torch.matmul(probs, v)


def standard_attn(q, k, v, causal=False):
    """Reference FP16 standard attention."""
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        N = q.shape[-2]
        s.masked_fill_(torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1), float('-inf'))
    return torch.matmul(torch.softmax(s, dim=-1), v.float()).to(q.dtype)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)

    # typical diffusion model: 64×64 image, 8×8 patches → 64 tokens; D=64
    # or stable diffusion xl: 128×128 latent/4 = 1024 tokens
    B, H, N, D = 2, 8, 1024, 64
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    out_sage = sage_attn_pytorch(q, k, v)
    out_ref  = standard_attn(q, k, v)
    diff = (out_sage.float() - out_ref.float()).abs().max().item()
    print(f"sage_attn (pytorch)  max|sage-ref|={diff:.5f}  {'OK (INT8 quant noise expected)' if diff < 0.1 else 'FAIL'}")

    # memory comparison: INT8 QK halves QK bandwidth (N²×1B vs N²×2B)
    qk_bandwidth_fp16_mb = N * N * 2 / 1e6
    qk_bandwidth_int8_mb = N * N * 1 / 1e6
    print(f"  QK bandwidth (N={N}): FP16={qk_bandwidth_fp16_mb:.1f}MB  INT8={qk_bandwidth_int8_mb:.1f}MB  (2× saving)")

    for _ in range(10): sage_attn_pytorch(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): sage_attn_pytorch(q, k, v)
    torch.cuda.synchronize()
    ms_sage = (time.perf_counter() - t0) / 100 * 1e3

    for _ in range(10): standard_attn(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): standard_attn(q, k, v)
    torch.cuda.synchronize()
    ms_ref = (time.perf_counter() - t0) / 100 * 1e3

    print(f"  sage={ms_sage:.3f}ms  standard={ms_ref:.3f}ms  (B={B} H={H} N={N} D={D})")
