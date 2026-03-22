import math
import torch
import triton
import triton.language as tl

# SM86 (RTX 3060): 28 SMs, 65536 regs/SM, 48KB shmem/SM, HMMA.16816 Tensor Cores
# tl.dot → HMMA.16816: 128 FP16 FMAs/warp/cycle, 4 warps/block → 512 FP16 FMAs/cycle
# BLOCK_M=64, BLOCK_N=64: 64×64×2B = 8KB FP16 output tile; reduction over BLOCK_K=32

@triton.jit
def _qkt_kernel(
    Q_ptr, K_ptr, S_ptr,
    N, D: tl.constexpr,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_sbh, stride_sn, stride_sm,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid (B*H, M_tiles, N_tiles): BLOCK_M×BLOCK_K Q-tile + BLOCK_N×BLOCK_K K-tile → 2×64×32×2B = 8KB SRAM
    # LDS.128 per warp per K-step: (BLOCK_M + BLOCK_N) × BLOCK_K × 2B / 32 threads = 256B/warp
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    Q_bh = Q_ptr + pid_bh * stride_qbh
    K_bh = K_ptr + pid_bh * stride_kbh
    S_bh = S_ptr + pid_bh * stride_sbh

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, D, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        q = tl.load(Q_bh + m_offs[:, None] * stride_qn + k_offs[None, :] * stride_qd,
                    mask=(m_offs[:, None] < N) & (k_offs[None, :] < D), other=0.0).to(tl.float32)
        k = tl.load(K_bh + n_offs[:, None] * stride_kn + k_offs[None, :] * stride_kd,
                    mask=(n_offs[:, None] < N) & (k_offs[None, :] < D), other=0.0).to(tl.float32)

        # HMMA.16816 via tl.dot; acc in fp32 avoids overflow for large D
        acc = tl.dot(q, tl.trans(k), acc)

    acc = acc * scale
    s_ptrs = S_bh + m_offs[:, None] * stride_sn + n_offs[None, :] * stride_sm
    tl.store(s_ptrs, acc, mask=(m_offs[:, None] < N) & (n_offs[None, :] < N))


@triton.jit
def _sv_kernel(
    P_ptr, V_ptr, Out_ptr,
    N, D: tl.constexpr,
    stride_pbh, stride_pn, stride_pm,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # BLOCK_M×BLOCK_N prob-tile (N×N fp32) + BLOCK_N×BLOCK_D V-tile: (64×64×4 + 64×64×2)B = 24KB SRAM
    # at 4 warps/block: 24KB / 48KB → 2 concurrent blocks/SM on shmem bound (16 warps, 25% warp occ)
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    pid_d  = tl.program_id(2)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    P_bh   = P_ptr   + pid_bh * stride_pbh
    V_bh   = V_ptr   + pid_bh * stride_vbh
    Out_bh = Out_ptr + pid_bh * stride_obh

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)

        p = tl.load(P_bh + m_offs[:, None] * stride_pn + n_offs[None, :] * stride_pm,
                    mask=(m_offs[:, None] < N) & (n_offs[None, :] < N), other=0.0).to(tl.float32)
        v = tl.load(V_bh + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vd,
                    mask=(n_offs[:, None] < N) & (d_offs[None, :] < D), other=0.0).to(tl.float32)

        acc = tl.dot(p, v, acc)

    tl.store(Out_bh + m_offs[:, None] * stride_on + d_offs[None, :] * stride_od,
             acc.to(tl.float16),
             mask=(m_offs[:, None] < N) & (d_offs[None, :] < D))


def triton_mha(q, k, v, causal=False):
    B, H, N, D = q.shape
    assert q.dtype == torch.float16 and q.is_cuda
    scale = 1.0 / math.sqrt(D)

    q_bhn = q.reshape(B * H, N, D).contiguous()
    k_bhn = k.reshape(B * H, N, D).contiguous()
    v_bhn = v.reshape(B * H, N, D).contiguous()

    # N×N fp32 score matrix: B*H * N² * 4B — the memory cost flash attention eliminates
    scores = torch.empty(B * H, N, N, device=q.device, dtype=torch.float32)
    out    = torch.empty_like(q_bhn)

    BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_D = 64, 64, 32, 64

    _qkt_kernel[(B * H, triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        q_bhn, k_bhn, scores, N, D,
        q_bhn.stride(0), q_bhn.stride(1), q_bhn.stride(2),
        k_bhn.stride(0), k_bhn.stride(1), k_bhn.stride(2),
        scores.stride(0), scores.stride(1), scores.stride(2),
        scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    if causal:
        mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0), float('-inf'))

    probs = torch.softmax(scores, dim=-1)

    _sv_kernel[(B * H, triton.cdiv(N, BLOCK_M), triton.cdiv(D, BLOCK_D))](
        probs, v_bhn, out, N, D,
        probs.stride(0), probs.stride(1), probs.stride(2),
        v_bhn.stride(0), v_bhn.stride(1), v_bhn.stride(2),
        out.stride(0),   out.stride(1),   out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return out.reshape(B, H, N, D)


def reference_mha(q, k, v, causal=False):
    D = q.shape[-1]
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(D)
    if causal:
        N = q.shape[-2]
        s.masked_fill_(torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1), float('-inf'))
    return torch.matmul(torch.softmax(s, dim=-1), v.float()).to(q.dtype)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 512, 64
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    out_t = triton_mha(q, k, v)
    out_r = reference_mha(q, k, v)
    diff  = (out_t.float() - out_r.float()).abs().max().item()
    print(f"fp16_mha_tiled_sm86  max|triton-ref|={diff:.5f}  {'OK' if diff < 0.05 else 'FAIL'}")

    for _ in range(20): triton_mha(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200): triton_mha(q, k, v)
    torch.cuda.synchronize()
    print(f"  {(time.perf_counter()-t0)/200*1e3:.3f} ms/call  (B={B} H={H} N={N} D={D})")
