import math
import torch
import triton
import triton.language as tl

# StreamingLLM attention: attend only to attention-sink tokens + sliding window
# KV memory: (sink_size + window_size) × D × 2 × 2B FP16 (K+V) = constant regardless of N
# vs standard: N × D × 4B grows unbounded; at N=100K, D=128: standard=100MB, streaming=0.5MB
# SM86: BLOCK_Q=32, BLOCK_KV=64 → Q-tile 32×64×2B=4KB + KV-tile 64×64×2B=8KB = 12KB SRAM
# HBM traffic: (sink_size + window_size) × D × 4B per query tile vs N × D × 4B standard

@triton.jit
def _streaming_attn_fwd(
    Q_ptr, K_sink_ptr, V_sink_ptr, K_win_ptr, V_win_ptr, Out_ptr,
    # strides for Q, K_sink, V_sink shaped (B*H, N_q, D)
    # K_win, V_win shaped (B*H, window_size, D)
    stride_qbh, stride_qn, stride_qd,
    stride_skbh, stride_skn, stride_skd,
    stride_wkbh, stride_wkn, stride_wkd,
    stride_obh, stride_on, stride_od,
    N_q, sink_size, window_size,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q:  tl.constexpr,
    BLOCK_KV: tl.constexpr,
    scale,
):
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)

    q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, HEAD_DIM)

    Q_bh    = Q_ptr     + pid_bh * stride_qbh
    Ks_bh   = K_sink_ptr + pid_bh * stride_skbh
    Vs_bh   = V_sink_ptr + pid_bh * stride_skbh
    Kw_bh   = K_win_ptr  + pid_bh * stride_wkbh
    Vw_bh   = V_win_ptr  + pid_bh * stride_wkbh
    Out_bh  = Out_ptr    + pid_bh * stride_obh

    q = tl.load(Q_bh + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
                mask=q_offs[:, None] < N_q, other=0.0)

    m_i = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # --- sink tokens: always attend (attention sinks absorb global context) ---
    # sink_size typically 4; these tokens see disproportionate attention weight
    # bandwidth: sink_size × D × 2B = sink_size × 128B → negligible vs window
    for kv in range(tl.cdiv(sink_size, BLOCK_KV)):
        kv_offs = kv * BLOCK_KV + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs[:, None] < sink_size

        k = tl.load(Ks_bh + kv_offs[:, None] * stride_skn + d_offs[None, :] * stride_skd,
                    mask=kv_mask, other=0.0)
        v = tl.load(Vs_bh + kv_offs[:, None] * stride_skn + d_offs[None, :] * stride_skd,
                    mask=kv_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        scores = tl.where(kv_offs[None, :] < sink_size, scores, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        o_i   = o_i * alpha[:, None] + tl.dot(p.to(tl.float16), v).to(tl.float32)
        m_i   = m_new

    # --- sliding window: most recent window_size tokens ---
    # window_size=512 → 512 × 128 × 2B K + V = 256KB total KV DRAM read per query block
    for kv in range(tl.cdiv(window_size, BLOCK_KV)):
        kv_offs = kv * BLOCK_KV + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs[:, None] < window_size

        k = tl.load(Kw_bh + kv_offs[:, None] * stride_wkn + d_offs[None, :] * stride_wkd,
                    mask=kv_mask, other=0.0)
        v = tl.load(Vw_bh + kv_offs[:, None] * stride_wkn + d_offs[None, :] * stride_wkd,
                    mask=kv_mask, other=0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        scores = tl.where(kv_offs[None, :] < window_size, scores, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        o_i   = o_i * alpha[:, None] + tl.dot(p.to(tl.float16), v).to(tl.float32)
        m_i   = m_new

    o_i = o_i / l_i[:, None]
    tl.store(Out_bh + q_offs[:, None] * stride_on + d_offs[None, :] * stride_od,
             o_i.to(tl.float16), mask=q_offs[:, None] < N_q)


def streaming_attn(q, k, v, sink_size=4, window_size=512):
    """Streaming attention with attention sinks and sliding window KV cache.

    q: (B, H, N_q, D) — current query tokens (can be single token decode)
    k, v: (B, H, N_kv, D) — full KV context; we slice sink + most-recent window
    """
    B, H, N_q, D = q.shape
    assert D in (32, 64, 128)
    assert q.dtype == torch.float16

    N_kv   = k.shape[2]
    actual_sink   = min(sink_size, N_kv)
    actual_window = min(window_size, max(0, N_kv - actual_sink))

    k_sink = k[:, :, :actual_sink, :].reshape(B * H, actual_sink, D).contiguous()
    v_sink = v[:, :, :actual_sink, :].reshape(B * H, actual_sink, D).contiguous()
    k_win  = k[:, :, N_kv - actual_window:, :].reshape(B * H, actual_window, D).contiguous() if actual_window > 0 \
             else torch.zeros(B * H, 1, D, device=q.device, dtype=q.dtype)
    v_win  = v[:, :, N_kv - actual_window:, :].reshape(B * H, actual_window, D).contiguous() if actual_window > 0 \
             else torch.zeros(B * H, 1, D, device=q.device, dtype=q.dtype)
    win_sz = actual_window if actual_window > 0 else 0

    q_bhn = q.reshape(B * H, N_q, D).contiguous()
    out   = torch.empty_like(q_bhn)

    BLOCK_Q, BLOCK_KV = 32, 64

    _streaming_attn_fwd[(B * H, triton.cdiv(N_q, BLOCK_Q))](
        q_bhn, k_sink, v_sink, k_win, v_win, out,
        q_bhn.stride(0), q_bhn.stride(1), q_bhn.stride(2),
        k_sink.stride(0), k_sink.stride(1), k_sink.stride(2),
        k_win.stride(0),  k_win.stride(1),  k_win.stride(2),
        out.stride(0),    out.stride(1),    out.stride(2),
        N_q=N_q, sink_size=actual_sink, window_size=max(win_sz, 1),
        HEAD_DIM=D, BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
        scale=1.0 / math.sqrt(D),
        num_warps=4,
    )
    return out.reshape(B, H, N_q, D)


def reference_attn(q, k, v):
    s = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.matmul(torch.softmax(s, dim=-1), v.float()).to(q.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 1024, 64
    SINK, WINDOW = 4, 256

    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    out_s = streaming_attn(q, k, v, sink_size=SINK, window_size=WINDOW)

    # reference: attend only to sink + window slice
    k_reduced = torch.cat([k[:, :, :SINK, :], k[:, :, N-WINDOW:, :]], dim=2)
    v_reduced = torch.cat([v[:, :, :SINK, :], v[:, :, N-WINDOW:, :]], dim=2)
    out_r = reference_attn(q, k_reduced, v_reduced)

    diff = (out_s.float() - out_r.float()).abs().max().item()
    print(f"fp16_streaming_attn_sm86  max|triton-ref|={diff:.5f}  {'OK' if diff < 0.05 else 'FAIL'}")

    kv_mem_standard  = N * D * 2 * 2 * B * H / 1e6
    kv_mem_streaming = (SINK + WINDOW) * D * 2 * 2 * B * H / 1e6
    print(f"  KV memory: standard={kv_mem_standard:.1f}MB  streaming={kv_mem_streaming:.1f}MB  "
          f"({(1-kv_mem_streaming/kv_mem_standard)*100:.0f}% reduction)")
