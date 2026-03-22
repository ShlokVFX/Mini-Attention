import math
import torch
from torch.utils.cpp_extension import load_inline

# SM86 (RTX 3060): 48KB shmem/SM, 65536 regs/SM, 28 SMs
# BLOCK_Q=32, BLOCK_KV=32: K_tile+V_tile = 2×32×D×4B = 16KB SRAM → 3 concurrent blocks/SM on shmem (48KB/16KB)
# register pressure: q[128] + o_i[128] + m_i,l_i = ~260 fp32 regs/thread at D=64
#   32 threads × 260 regs = 8320 regs/block; 65536/8320 ≈ 7 blocks/SM → 7×32=224 warps, but shmem caps at 3
# arithmetic intensity (flash): 4×N×D×B HBM reads → 2×N²×D FLOPs → N/2 FLOPs/byte
#   N=1024, D=64: 512 FLOPs/byte >> 35 FLOPs/byte 3060 roofline crossover → compute-bound at N>70

_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_Q  32
#define BLOCK_KV 32
// max supported HEAD_DIM; increase if needed (costs more registers)
#define MAX_D 128

// Grid (B, H, Q_tiles): one block per query tile
// Block: BLOCK_Q threads; each thread owns one query row end-to-end
// Shmem: K_tile[BLOCK_KV×D] + V_tile[BLOCK_KV×D] = 2×32×D×4B; D=64 → 16KB/block
__global__ void flash_fwd(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ Out,
    int B, int H, int N, int D,
    float scale, bool causal)
{
    int b = blockIdx.x, h = blockIdx.y, q_tile = blockIdx.z;
    int tid = threadIdx.x;
    int q_global = q_tile * BLOCK_Q + tid;

    int bh_off = (b * H + h) * N * D;

    // K_tile and V_tile in shmem; each tile loaded cooperatively by BLOCK_Q threads
    extern __shared__ float tiles[];
    float* K_tile = tiles;
    float* V_tile = tiles + BLOCK_KV * D;

    // per-thread query + accumulator in registers — no shmem needed for Q or output
    // at D=64: q[64]+o[64]+m+l = 130 fp32 regs/thread; 32 threads → 4160 regs/block (6.3% of SM86)
    float q[MAX_D], o_i[MAX_D];
    float m_i = -1e20f, l_i = 0.0f;

    for (int d = 0; d < D; d++) {
        q[d]  = (q_global < N) ? Q[bh_off + q_global * D + d] : 0.0f;
        o_i[d] = 0.0f;
    }

    int n_kv = (N + BLOCK_KV - 1) / BLOCK_KV;
    for (int kv = 0; kv < n_kv; kv++) {
        int kv_start = kv * BLOCK_KV;

        // cooperative tile load: BLOCK_Q threads fill BLOCK_KV×D elements
        // LDS.128 (16B/load) would require aligned D; using scalar loads here for clarity
        for (int i = tid; i < BLOCK_KV * D; i += BLOCK_Q) {
            int row = i / D, col = i % D;
            int g   = kv_start + row;
            K_tile[i] = (g < N) ? K[bh_off + g * D + col] : 0.0f;
            V_tile[i] = (g < N) ? V[bh_off + g * D + col] : 0.0f;
        }
        __syncthreads();

        if (q_global < N) {
            for (int j = 0; j < BLOCK_KV; j++) {
                int kv_global = kv_start + j;
                if (kv_global >= N) break;
                if (causal && kv_global > q_global) break;

                float score = 0.0f;
                for (int d = 0; d < D; d++) score += q[d] * K_tile[j * D + d];
                score *= scale;

                // online softmax: rescale running accumulator, no N² buffer needed
                float m_new = max(m_i, score);
                float alpha = expf(m_i - m_new);
                float p     = expf(score - m_new);
                l_i = l_i * alpha + p;
                for (int d = 0; d < D; d++)
                    o_i[d] = o_i[d] * alpha + p * V_tile[j * D + d];
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (q_global < N) {
        for (int d = 0; d < D; d++)
            Out[bh_off + q_global * D + d] = o_i[d] / l_i;
    }
}

torch::Tensor flash_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal)
{
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat32);
    TORCH_CHECK(Q.size(3) <= MAX_D, "head_dim exceeds MAX_D=128");
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    auto Out = torch::empty_like(Q);
    // shmem: 2 × BLOCK_KV × D × 4B
    size_t shmem = 2 * BLOCK_KV * D * sizeof(float);
    dim3 grid(B, H, (N + BLOCK_Q - 1) / BLOCK_Q);
    flash_fwd<<<grid, BLOCK_Q, shmem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        Out.data_ptr<float>(), B, H, N, D, 1.0f/sqrtf(D), causal);
    return Out;
}
"""

_ext = load_inline(
    name="fp32_flash_attn_sm86",
    cpp_sources="torch::Tensor flash_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal);",
    cuda_sources=_CUDA,
    functions=["flash_attn_forward"],
    extra_cuda_cflags=["-O2", "--use_fast_math", "-arch=sm_86"],
    verbose=False,
)

def flash_attn(q, k, v, causal=False):
    return _ext.flash_attn_forward(q.contiguous(), k.contiguous(), v.contiguous(), causal)

def reference(q, k, v, causal=False):
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        N = q.shape[-2]
        s.masked_fill_(torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1), float('-inf'))
    return torch.matmul(torch.softmax(s, dim=-1), v)

if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 1024, 64
    q = torch.randn(B, H, N, D, device='cuda')
    k = torch.randn(B, H, N, D, device='cuda')
    v = torch.randn(B, H, N, D, device='cuda')

    for causal in [False, True]:
        diff = (flash_attn(q, k, v, causal) - reference(q, k, v, causal)).abs().max().item()
        print(f"fp32_flash_attn_sm86  causal={causal}  max|fa-ref|={diff:.6f}  {'OK' if diff < 1e-4 else 'FAIL'}")

    torch.cuda.reset_peak_memory_stats()
    flash_attn(q, k, v)
    torch.cuda.synchronize()
    mem_fa = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    reference(q, k, v)
    torch.cuda.synchronize()
    mem_ref = torch.cuda.max_memory_allocated() / 1e6
    print(f"  peak mem: flash={mem_fa:.1f}MB  standard={mem_ref:.1f}MB")

    for _ in range(10): flash_attn(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): flash_attn(q, k, v)
    torch.cuda.synchronize()
    print(f"  {(time.perf_counter()-t0)/100*1e3:.3f} ms/call")
