import math
import torch
import triton
import triton.language as tl

# Paged Attention: K/V stored in non-contiguous fixed-size pages (like virtual memory)
# block_table[seq, block_idx] → physical page ID; indirect addressing adds 1 pointer lookup per K/V tile
# page_size=16: each page = 16 × D × 2 × 2B = 16×64×4B = 4KB (K+V); fits cleanly in SM86 cache line
# memory fragmentation: standard KV = up to 50% waste from pre-allocation; paged = ~4% (one partial last page)
# SM86: BLOCK_Q=32, page_size=16 → 2 pages per Q tile KV step; block_table indirection cost: 1 LDS per page

@triton.jit
def _paged_attn_fwd(
    Q_ptr,          # (B, H, N_q, D)
    K_pages_ptr,    # (num_pages, page_size, D) — physical K page pool
    V_pages_ptr,    # (num_pages, page_size, D)
    block_table_ptr, # (B, max_blocks) int32 — logical → physical page mapping
    Out_ptr,
    # strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kp, stride_kps, stride_kd,   # K_pages: (page, slot, dim)
    stride_ob, stride_oh, stride_on, stride_od,
    stride_bt_b, stride_bt_blk,
    N_q, N_kv, D: tl.constexpr,
    page_size: tl.constexpr,
    max_blocks: tl.constexpr,
    scale,
    BLOCK_Q: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, D)

    q = tl.load(Q_ptr + pid_b * stride_qb + pid_h * stride_qh
                + q_offs[:, None] * stride_qn + d_offs[None, :] * stride_qd,
                mask=q_offs[:, None] < N_q, other=0.0)

    m_i = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_Q, D], dtype=tl.float32)

    # iterate over logical blocks; each block_table lookup costs one L1 hit (4B pointer read)
    # HBM traffic: N_kv/page_size pointer lookups + N_kv × D × 2B K data + N_kv × D × 2B V data
    n_logical_blocks = tl.cdiv(N_kv, page_size)

    for blk_idx in range(n_logical_blocks):
        # indirect: read physical page ID from block_table (4B; usually L1-cached after first access)
        phys_page = tl.load(block_table_ptr + pid_b * stride_bt_b + blk_idx * stride_bt_blk)

        slot_offs = tl.arange(0, page_size)  # slots within this page
        kv_offs   = blk_idx * page_size + slot_offs
        kv_mask   = kv_offs < N_kv

        # K/V at physical address: phys_page × page_size × D
        k = tl.load(K_pages_ptr + phys_page * stride_kp + slot_offs[:, None] * stride_kps + d_offs[None, :] * stride_kd,
                    mask=kv_mask[:, None], other=0.0)
        v = tl.load(V_pages_ptr + phys_page * stride_kp + slot_offs[:, None] * stride_kps + d_offs[None, :] * stride_kd,
                    mask=kv_mask[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        scores = tl.where(kv_offs[None, :] < N_kv, scores, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        o_i   = o_i * alpha[:, None] + tl.dot(p.to(tl.float16), v).to(tl.float32)
        m_i   = m_new

    o_i = o_i / l_i[:, None]
    tl.store(Out_ptr + pid_b * stride_ob + pid_h * stride_oh
             + q_offs[:, None] * stride_on + d_offs[None, :] * stride_od,
             o_i.to(tl.float16), mask=q_offs[:, None] < N_q)


def paged_attention(q, k_pages, v_pages, block_table, N_kv):
    """Paged flash attention.

    q:           (B, H, N_q, D) fp16
    k_pages:     (num_pages, page_size, D) fp16 — physical K page pool
    v_pages:     (num_pages, page_size, D) fp16
    block_table: (B, max_blocks) int32 — logical block → physical page
    N_kv:        int — actual KV sequence length
    """
    B, H, N_q, D = q.shape
    assert D in (32, 64, 128) and q.dtype == torch.float16
    page_size  = k_pages.shape[1]
    max_blocks = block_table.shape[1]

    out = torch.empty_like(q)
    BLOCK_Q = 32

    _paged_attn_fwd[(B, H, triton.cdiv(N_q, BLOCK_Q))](
        q, k_pages, v_pages, block_table, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_pages.stride(0), k_pages.stride(1), k_pages.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        block_table.stride(0), block_table.stride(1),
        N_q=N_q, N_kv=N_kv, D=D,
        page_size=page_size, max_blocks=max_blocks,
        scale=1.0 / math.sqrt(D),
        BLOCK_Q=BLOCK_Q, num_warps=4,
    )
    return out


def build_paged_kv(k, v, page_size=16):
    """Pack contiguous KV into page pool + block_table for testing."""
    B, H, N, D = k.shape
    n_pages_per_seq = (N + page_size - 1) // page_size
    total_pages = B * H * n_pages_per_seq

    k_pages = torch.zeros(total_pages, page_size, D, device=k.device, dtype=k.dtype)
    v_pages = torch.zeros(total_pages, page_size, D, device=k.device, dtype=k.dtype)
    block_table = torch.zeros(B * H, n_pages_per_seq, dtype=torch.int32, device=k.device)

    page_id = 0
    for bh in range(B * H):
        b, h = bh // H, bh % H
        for blk in range(n_pages_per_seq):
            start = blk * page_size
            end   = min(start + page_size, N)
            k_pages[page_id, :end - start, :] = k[b, h, start:end, :]
            v_pages[page_id, :end - start, :] = v[b, h, start:end, :]
            block_table[bh, blk] = page_id
            page_id += 1

    return k_pages, v_pages, block_table


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 512, 64
    PAGE_SIZE = 16

    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    k_pages, v_pages, block_table = build_paged_kv(k, v, PAGE_SIZE)

    # paged attention expects (B*H, N, D) query
    q_bh = q.reshape(B * H, 1, N, D)  # treat each head independently

    # reshape for the kernel: (B*H, 1 head, N_q, D)
    q2 = q.reshape(B * H, 1, N, D)
    k_pages2 = k_pages
    v_pages2 = v_pages
    bt2 = block_table

    out_paged = paged_attention(q2, k_pages2, v_pages2, bt2, N_kv=N)

    # reference standard attention
    s   = torch.matmul(q.float().reshape(B*H, N, D), k.float().reshape(B*H, N, D).transpose(-2, -1)) / math.sqrt(D)
    ref = torch.matmul(torch.softmax(s, dim=-1), v.float().reshape(B*H, N, D))
    ref = ref.reshape(B, H, N, D).to(torch.float16)

    diff = (out_paged.reshape(B, H, N, D).float() - ref.float()).abs().max().item()
    print(f"fp16_paged_attn_sm86  max|paged-ref|={diff:.5f}  {'OK' if diff < 0.05 else 'FAIL'}")

    contiguous_kv_mb = B * H * N * D * 2 * 2 / 1e6
    paged_kv_mb      = k_pages.numel() * 2 * 2 / 1e6
    print(f"  KV storage: contiguous={contiguous_kv_mb:.1f}MB  paged={paged_kv_mb:.1f}MB  "
          f"(page overhead={(paged_kv_mb/contiguous_kv_mb-1)*100:.1f}%)")
