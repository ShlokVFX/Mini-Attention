import math
import torch
import triton
import triton.language as tl

# SM86 (RTX 3060): HBM 360 GB/s, L1/shmem 48KB/SM, 65536 regs/SM
# Flash Attention 2: O(N·D) HBM reads vs O(N²) standard — arithmetic intensity = N/2 FLOPs/byte
# At N=1024, D=64: intensity=512 FLOPs/byte >> roofline crossover ~35 FLOPs/byte → compute-bound
# BLOCK_Q=64, BLOCK_KV=64: Q-tile 64×64×2B=8KB + K-tile 8KB + V-tile 8KB = 24KB SRAM → 2 blocks/SM
# num_stages=2: cp.async double-buffering hides ~100ns DRAM latency (SM86 L2 miss)
# 4 warps/block × 2 blocks/SM = 8 warps active → 12.5% SM86 warp occupancy (latency hidden by memory pipeline)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q':  64, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q':  64, 'BLOCK_KV': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_Q':  32, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
    ],
    key=['N', 'HEAD_DIM'],
)
@triton.jit
def _flash_fwd(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    N, HEAD_DIM: tl.constexpr,
    scale, causal: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
):
    # register budget: m_i[BLOCK_Q] + l_i[BLOCK_Q] + o_i[BLOCK_Q×HEAD_DIM] = 64+64+4096 fp32 regs
    # at BLOCK_Q=64, HEAD_DIM=64: 4224 regs/program; SM86 max 65536 → ~15 concurrent programs/SM
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)
    q_start = pid_q * BLOCK_Q

    q_offs = q_start + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, HEAD_DIM)

    Q_bh   = Q_ptr   + pid_bh * stride_qbh
    K_bh   = K_ptr   + pid_bh * stride_kbh
    V_bh   = V_ptr   + pid_bh * stride_vbh
    Out_bh = Out_ptr + pid_bh * stride_obh

    # Q tile loaded once into SRAM; K/V tiles streamed from HBM each iteration
    q = tl.load(Q_bh + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
                mask=q_offs[:, None] < N, other=0.0)

    m_i = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    if causal:
        n_kv = tl.cdiv((pid_q + 1) * BLOCK_Q, BLOCK_KV)
    else:
        n_kv = tl.cdiv(N, BLOCK_KV)

    for kv in range(n_kv):
        kv_start = kv * BLOCK_KV
        kv_offs  = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask  = kv_offs[:, None] < N

        k = tl.load(K_bh + kv_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
                    mask=kv_mask, other=0.0)
        v = tl.load(V_bh + kv_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
                    mask=kv_mask, other=0.0)

        # HMMA.16816: tl.dot on fp16 inputs → fp32 accumulator; BLOCK_Q×BLOCK_KV tile = 64×64 FP32 = 16KB
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        scores = tl.where(kv_offs[None, :] < N, scores, float('-inf'))

        if causal:
            scores = tl.where(q_offs[:, None] >= kv_offs[None, :], scores, float('-inf'))

        # online softmax: 2 passes over BLOCK_KV scores — avoids storing N² matrix
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_i = o_i * alpha[:, None] + tl.dot(p.to(tl.float16), v).to(tl.float32)
        m_i = m_new

    o_i = o_i / l_i[:, None]
    tl.store(Out_bh + q_offs[:, None] * stride_on + d_offs[None, :] * stride_od,
             o_i.to(tl.float16), mask=q_offs[:, None] < N)


def flash_attn(q, k, v, causal=False):
    B, H, N, D = q.shape
    assert q.dtype == torch.float16 and q.is_cuda
    assert D in (32, 64, 128)

    q_bhn = q.reshape(B * H, N, D).contiguous()
    k_bhn = k.reshape(B * H, N, D).contiguous()
    v_bhn = v.reshape(B * H, N, D).contiguous()
    out   = torch.empty_like(q_bhn)

    _flash_fwd[lambda meta: (B * H, triton.cdiv(N, meta['BLOCK_Q']))](
        q_bhn, k_bhn, v_bhn, out,
        q_bhn.stride(0), q_bhn.stride(1), q_bhn.stride(2),
        k_bhn.stride(0), k_bhn.stride(1), k_bhn.stride(2),
        v_bhn.stride(0), v_bhn.stride(1), v_bhn.stride(2),
        out.stride(0),   out.stride(1),   out.stride(2),
        N=N, HEAD_DIM=D, scale=1.0/math.sqrt(D), causal=causal,
    )
    return out.reshape(B, H, N, D)


def reference(q, k, v, causal=False):
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        N = q.shape[-2]
        s.masked_fill_(torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1), float('-inf'))
    return torch.matmul(torch.softmax(s, dim=-1), v.float()).to(q.dtype)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 1024, 64
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    for causal in [False, True]:
        diff = (flash_attn(q, k, v, causal).float() - reference(q, k, v, causal).float()).abs().max().item()
        print(f"fp16_flash_attn_sm86  causal={causal}  max|fa-ref|={diff:.5f}  {'OK' if diff < 0.05 else 'FAIL'}")

    torch.cuda.reset_peak_memory_stats()
    flash_attn(q, k, v)
    torch.cuda.synchronize()
    mem_fa = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    reference(q, k, v)
    torch.cuda.synchronize()
    mem_ref = torch.cuda.max_memory_allocated() / 1e6
    print(f"  peak mem: flash={mem_fa:.1f}MB  standard={mem_ref:.1f}MB  ({(1-mem_fa/mem_ref)*100:.0f}% reduction)")

    for _ in range(20): flash_attn(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200): flash_attn(q, k, v)
    torch.cuda.synchronize()
    print(f"  {(time.perf_counter()-t0)/200*1e3:.3f} ms/call  (B={B} H={H} N={N} D={D})")
