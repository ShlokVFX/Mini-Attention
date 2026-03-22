import math
import torch
from torch.utils.cpp_extension import load_inline

# SM86 (RTX 3060): 48KB shmem/SM, 65536 regs/SM, 360 GB/s HBM
# v3 optimizations vs v2 (BLOCK_Q=32, BLOCK_KV=64, fp16 shmem, scalar KV HBM loads, scalar half→float shmem reads):
#   1. float2 HBM→half2 shmem tile fill: 64-bit loads instead of 32-bit → 2× fewer load instructions (64 iters vs 128)
#      __float22half2_rn packs two fp32→fp16 in one instruction; stores as half2 (32-bit write vs two 16-bit writes)
#   2. half2 shmem reads in QK^T dot-product: D/2=32 half2 loads per j instead of D=64 scalar half loads → 2× fewer SMEM ops
#   3. half2 shmem reads in PV accumulate: same gain — 32 half2 loads per j instead of 64 scalar loads
# register budget: q[64]+o[64]+m+l = 130 fp32/thread; 32 threads × 130 = 4160 regs/block (unchanged from v2, no spill)
# shmem: 2×64×64×2B = 16KB (unchanged) → 3 concurrent blocks/SM on RTX 3060

_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define BLOCK_Q  32    // threads/block = 1 warp; each thread owns one query row
#define BLOCK_KV 64    // same as v2; enabled by fp16 shmem
#define MAX_D    64    // head_dim; fixed for this kernel

// Grid (B, H, Q_tiles); Block = BLOCK_Q threads = 1 warp
// Shmem: K_half[BLOCK_KV×D] fp16 + V_half[BLOCK_KV×D] fp16 = 16KB → 3 blocks/SM
// v3 tile load: float2 HBM reads → __float22half2_rn → half2 shmem stores
//   BLOCK_KV×D = 4096 fp16 = 2048 half2; 32 threads × 64 half2 iters (vs 128 scalar in v2)
// v3 inner loop: half2 shmem reads → D/2=32 half2 loads per KV pos (vs D=64 scalar in v2)
__global__ void flash_fwd_v3(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ Out,
    int B, int H, int N, int D,
    float scale, bool causal)
{
    extern __shared__ __align__(16) char smem_raw[];
    __half*  K_half  = reinterpret_cast<__half*>(smem_raw);
    __half*  V_half  = K_half + BLOCK_KV * MAX_D;
    // half2 aliases for vectorized shmem access (no extra memory — same backing array)
    __half2* K_half2 = reinterpret_cast<__half2*>(K_half);
    __half2* V_half2 = reinterpret_cast<__half2*>(V_half);

    int b = blockIdx.x, h = blockIdx.y, q_tile = blockIdx.z;
    int tid = threadIdx.x;
    int q_global = q_tile * BLOCK_Q + tid;
    int bh_off = (b * H + h) * N * D;

    // Q row in fp32 registers (float4 load: 16 loads/row, same as v2)
    float q[MAX_D], o_i[MAX_D];
    float m_i = -1e20f, l_i = 0.0f;

    if (q_global < N) {
        const float4* q_row4 = reinterpret_cast<const float4*>(Q + bh_off + q_global * D);
        #pragma unroll
        for (int d4 = 0; d4 < D / 4; d4++) {
            float4 v = q_row4[d4];
            q[d4*4+0]=v.x; q[d4*4+1]=v.y; q[d4*4+2]=v.z; q[d4*4+3]=v.w;
        }
    } else {
        #pragma unroll
        for (int d = 0; d < MAX_D; d++) q[d] = 0.0f;
    }
    #pragma unroll
    for (int d = 0; d < MAX_D; d++) o_i[d] = 0.0f;

    int n_kv = (N + BLOCK_KV - 1) / BLOCK_KV;
    for (int kv = 0; kv < n_kv; kv++) {
        int kv_base = kv * BLOCK_KV;

        // float2 HBM→half2 shmem: 2× wider than scalar fp32 loads in v2
        // D=64 (even), so base=i*2 → col=base%64 is always even: both floats are in the same K/V row
        // BLOCK_KV*D/2 = 2048 half2 elements; 32 threads → 64 iterations each (vs 128 scalar in v2)
        for (int i = tid; i < BLOCK_KV * D / 2; i += BLOCK_Q) {
            int base = i * 2;
            int row  = base / D, col = base % D;  // col always even, never crosses row boundary
            int g    = kv_base + row;
            if (g < N) {
                float2 kf = *reinterpret_cast<const float2*>(&K[bh_off + g * D + col]);
                float2 vf = *reinterpret_cast<const float2*>(&V[bh_off + g * D + col]);
                K_half2[i] = __float22half2_rn(kf);
                V_half2[i] = __float22half2_rn(vf);
            } else {
                K_half2[i] = __float22half2_rn(make_float2(0.f, 0.f));
                V_half2[i] = __float22half2_rn(make_float2(0.f, 0.f));
            }
        }
        __syncthreads();

        if (q_global < N) {
            // half2 dot-product: D/2=32 half2 shmem loads per j (vs D=64 scalar loads in v2)
            #pragma unroll 4
            for (int j = 0; j < BLOCK_KV; j++) {
                int kv_global = kv_base + j;
                if (kv_global >= N) break;
                if (causal && kv_global > q_global) break;

                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < MAX_D / 2; d++) {
                    float2 k2 = __half22float2(K_half2[j * (MAX_D / 2) + d]);
                    score += q[d*2] * k2.x + q[d*2+1] * k2.y;
                }
                score *= scale;

                float m_new = fmaxf(m_i, score);
                float alpha = expf(m_i - m_new);
                float p     = expf(score - m_new);
                l_i = l_i * alpha + p;

                // half2 PV accumulate: D/2=32 half2 shmem loads per j (vs D=64 scalar in v2)
                #pragma unroll
                for (int d = 0; d < MAX_D / 2; d++) {
                    float2 v2 = __half22float2(V_half2[j * (MAX_D / 2) + d]);
                    o_i[d*2]   = o_i[d*2]   * alpha + p * v2.x;
                    o_i[d*2+1] = o_i[d*2+1] * alpha + p * v2.y;
                }
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (q_global < N) {
        float inv_l = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;
        float4* out_row4 = reinterpret_cast<float4*>(Out + bh_off + q_global * D);
        #pragma unroll
        for (int d4 = 0; d4 < D / 4; d4++) {
            float4 v;
            v.x = o_i[d4*4+0] * inv_l; v.y = o_i[d4*4+1] * inv_l;
            v.z = o_i[d4*4+2] * inv_l; v.w = o_i[d4*4+3] * inv_l;
            out_row4[d4] = v;
        }
    }
}

torch::Tensor flash_attn_forward_v3(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal)
{
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat32);
    TORCH_CHECK(Q.size(3) == MAX_D, "v3 kernel requires head_dim == 64");
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    auto Out = torch::empty_like(Q);
    // shmem: 2 × BLOCK_KV × D × sizeof(__half) = 16KB (unchanged from v2)
    size_t shmem = 2 * BLOCK_KV * MAX_D * sizeof(__half);
    dim3 grid(B, H, (N + BLOCK_Q - 1) / BLOCK_Q);
    flash_fwd_v3<<<grid, BLOCK_Q, shmem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        Out.data_ptr<float>(), B, H, N, D, 1.0f / sqrtf((float)D), causal);
    return Out;
}
"""

_ext_v3 = load_inline(
    name="fp32_flash_attn_sm86_v3",
    cpp_sources="torch::Tensor flash_attn_forward_v3(torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal);",
    cuda_sources=_CUDA,
    functions=["flash_attn_forward_v3"],
    extra_cuda_cflags=["-O2", "--use_fast_math", "-arch=sm_86"],
    verbose=False,
)

def flash_attn_v3(q, k, v, causal=False):
    return _ext_v3.flash_attn_forward_v3(q.contiguous(), k.contiguous(), v.contiguous(), causal)

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
        diff = (flash_attn_v3(q, k, v, causal) - reference(q, k, v, causal)).abs().max().item()
        print(f"v3  causal={causal}  max|v3-ref|={diff:.6f}  {'OK' if diff < 1e-3 else 'FAIL'}")

    for _ in range(10): flash_attn_v3(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): flash_attn_v3(q, k, v)
    torch.cuda.synchronize()
    print(f"  v3: {(time.perf_counter()-t0)/100*1e3:.3f} ms/call  (B={B} H={H} N={N} D={D})")
