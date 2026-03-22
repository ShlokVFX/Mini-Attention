import math
import torch
from torch.utils.cpp_extension import load_inline

# Paged Attention CUDA: block_table[b*max_blocks + blk_idx] → physical page offset
# One extra global load per page (4B int32 pointer) vs contiguous KV — L1 cache hit after first access
# page_size=16: 16 × D × 4B = 4KB fp32 per K-page; 2 pages fit in 8KB L1 cache line on SM86
# shmem: K_page[page_size×D] + V_page[page_size×D] = 2×16×64×4B = 8KB → 6 blocks/SM on 48KB
# register use per thread: q[MAX_D=128] + o[MAX_D] + m,l = 258 fp32 → 258×32 = 8256 regs/block (12.5% of SM86)

_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_Q   32
#define PAGE_SIZE 16
#define MAX_D    128

// Grid (B*H, Q_tiles), Block BLOCK_Q threads
// Each thread owns one query row; follows block_table to find physical K/V pages
// Shmem: K_page[PAGE_SIZE×D] + V_page[PAGE_SIZE×D]; D=64 → 8KB per block
__global__ void paged_attn_fwd(
    const float*   __restrict__ Q,           // (B*H, N_q, D)
    const float*   __restrict__ K_pages,     // (num_pages, PAGE_SIZE, D)
    const float*   __restrict__ V_pages,
    const int*     __restrict__ block_table, // (B*H, max_blocks)
    float*         __restrict__ Out,
    int N_q, int N_kv, int D, int max_blocks,
    float scale)
{
    int bh     = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid    = threadIdx.x;
    int q_g    = q_tile * BLOCK_Q + tid;

    extern __shared__ float shmem[];
    float* K_pg = shmem;
    float* V_pg = shmem + PAGE_SIZE * D;

    float q[MAX_D], o_i[MAX_D];
    float m_i = -1e20f, l_i = 0.0f;
    for (int d = 0; d < D; d++) {
        q[d]   = (q_g < N_q) ? Q[bh * N_q * D + q_g * D + d] : 0.0f;
        o_i[d] = 0.0f;
    }

    int n_blocks = (N_kv + PAGE_SIZE - 1) / PAGE_SIZE;

    for (int blk = 0; blk < n_blocks; blk++) {
        // indirect page lookup — single 4B load; stays in L1 after first block
        int phys = block_table[bh * max_blocks + blk];
        int kv_start = blk * PAGE_SIZE;

        // cooperative tile load: BLOCK_Q threads fill PAGE_SIZE×D
        for (int i = tid; i < PAGE_SIZE * D; i += BLOCK_Q) {
            int row = i / D, col = i % D;
            int g   = kv_start + row;
            K_pg[i] = (g < N_kv) ? K_pages[phys * PAGE_SIZE * D + row * D + col] : 0.0f;
            V_pg[i] = (g < N_kv) ? V_pages[phys * PAGE_SIZE * D + row * D + col] : 0.0f;
        }
        __syncthreads();

        if (q_g < N_q) {
            for (int j = 0; j < PAGE_SIZE && kv_start + j < N_kv; j++) {
                float score = 0.0f;
                for (int d = 0; d < D; d++) score += q[d] * K_pg[j * D + d];
                score *= scale;

                float m_new = max(m_i, score);
                float alpha = expf(m_i - m_new), p = expf(score - m_new);
                l_i = l_i * alpha + p;
                for (int d = 0; d < D; d++)
                    o_i[d] = o_i[d] * alpha + p * V_pg[j * D + d];
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (q_g < N_q)
        for (int d = 0; d < D; d++)
            Out[bh * N_q * D + q_g * D + d] = o_i[d] / l_i;
}

torch::Tensor paged_attn_forward(
    torch::Tensor Q, torch::Tensor K_pages, torch::Tensor V_pages,
    torch::Tensor block_table, int N_kv)
{
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat32);
    TORCH_CHECK(Q.size(2) <= MAX_D, "head_dim > MAX_D");
    int BH = Q.size(0), N_q = Q.size(1), D = Q.size(2);
    int max_blocks = block_table.size(1);

    auto Out = torch::empty_like(Q);
    size_t shmem = 2 * PAGE_SIZE * D * sizeof(float);
    dim3 grid(BH, (N_q + BLOCK_Q - 1) / BLOCK_Q);

    paged_attn_fwd<<<grid, BLOCK_Q, shmem>>>(
        Q.data_ptr<float>(), K_pages.data_ptr<float>(), V_pages.data_ptr<float>(),
        block_table.data_ptr<int>(), Out.data_ptr<float>(),
        N_q, N_kv, D, max_blocks, 1.0f/sqrtf(D));
    return Out;
}
"""

_ext = load_inline(
    name="fp32_paged_attn_sm86",
    cpp_sources="""torch::Tensor paged_attn_forward(
        torch::Tensor Q, torch::Tensor K_pages, torch::Tensor V_pages,
        torch::Tensor block_table, int N_kv);""",
    cuda_sources=_CUDA,
    functions=["paged_attn_forward"],
    extra_cuda_cflags=["-O2", "--use_fast_math", "-arch=sm_86"],
    verbose=False,
)

PAGE_SIZE = 16

def build_paged_kv_fp32(k, v):
    """Pack (BH, N, D) contiguous KV into page pool + block_table."""
    BH, N, D = k.shape
    n_pages_per_seq = (N + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = BH * n_pages_per_seq

    k_pool = torch.zeros(total_pages, PAGE_SIZE, D, device=k.device)
    v_pool = torch.zeros(total_pages, PAGE_SIZE, D, device=k.device)
    bt     = torch.zeros(BH, n_pages_per_seq, dtype=torch.int32, device=k.device)

    pid = 0
    for bh in range(BH):
        for blk in range(n_pages_per_seq):
            s, e = blk * PAGE_SIZE, min((blk + 1) * PAGE_SIZE, N)
            k_pool[pid, :e-s] = k[bh, s:e]
            v_pool[pid, :e-s] = v[bh, s:e]
            bt[bh, blk] = pid
            pid += 1

    return k_pool, v_pool, bt


def paged_attn(q, k, v):
    """q, k, v: (BH, N, D) fp32."""
    k_pool, v_pool, bt = build_paged_kv_fp32(k, v)
    return _ext.paged_attn_forward(q.contiguous(), k_pool, v_pool, bt, k.shape[1])


if __name__ == "__main__":
    torch.manual_seed(0)
    BH, N, D = 16, 256, 64

    q = torch.randn(BH, N, D, device='cuda')
    k = torch.randn(BH, N, D, device='cuda')
    v = torch.randn(BH, N, D, device='cuda')

    out = paged_attn(q, k, v)
    s   = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    ref = torch.matmul(torch.softmax(s, dim=-1), v)

    diff = (out - ref).abs().max().item()
    print(f"fp32_paged_attn_sm86  max|paged-ref|={diff:.6f}  {'OK' if diff < 1e-4 else 'FAIL'}")
    print(f"  KV pool: {BH * N * D * 4 * 2 / 1e6:.1f}MB contiguous  "
          f"page overhead: {PAGE_SIZE - (N % PAGE_SIZE or PAGE_SIZE)} slots wasted/seq")
