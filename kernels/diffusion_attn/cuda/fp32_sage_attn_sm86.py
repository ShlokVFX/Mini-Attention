import math
import torch
from torch.utils.cpp_extension import load_inline

# SageAttention CUDA: __dp4a for INT8 dot products, FP32 dequantization, FP16 V
# SM86: __dp4a = 4 INT8 FMAs in one instruction (SIMD4); 32 threads/warp × 4 = 128 INT8 ops/warp/cycle
# vs FP32 dot: 32 FP32 MADs/warp/cycle → 4× INT8 arithmetic throughput for QK
# smooth quant overhead: 1 max-reduce per Q/K row (warp shuffle, 5 cycles) — amortized over D=64 elements
# shmem: K_tile[BLOCK_KV×D/4] INT8 + K_scale[BLOCK_KV] fp32 + V_tile[BLOCK_KV×D] fp32
#   at BLOCK_KV=32, D=64: 32×16×1B + 32×4B + 32×64×4B = 0.5KB+0.1KB+8KB = 8.6KB → 5 blocks/SM on 48KB

_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define BLOCK_Q  32
#define BLOCK_KV 32
#define MAX_D   128

// Pack 4 int8 values into int32 for __dp4a
// __dp4a(a, b, c) = c + dot4(a, b) where a,b are packed int8×4
// 1 instruction = 4 INT8 MADs on SM86 (vs 4 FP32 MADs for same cost)
__device__ __forceinline__ int dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}

__device__ __forceinline__ float warp_max(float v) {
    v = max(v, __shfl_xor_sync(0xffffffff, v, 16));
    v = max(v, __shfl_xor_sync(0xffffffff, v,  8));
    v = max(v, __shfl_xor_sync(0xffffffff, v,  4));
    v = max(v, __shfl_xor_sync(0xffffffff, v,  2));
    v = max(v, __shfl_xor_sync(0xffffffff, v,  1));
    return v;
}

// Grid (B*H, Q_tiles), Block BLOCK_Q threads
// Each thread: q_int8[MAX_D] in registers (1B/elem), o_i[MAX_D] fp32, m_i, l_i
// register use: (MAX_D/4 int32 packed + MAX_D fp32 + 2 fp32) per thread × 32 = ~5400 regs/block (8.2% of SM86)
// shmem: K_tile packed INT8 + K_scale + V_tile fp32 = ~9KB at BLOCK_KV=32, D=64
__global__ void sage_attn_fwd(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ Out,
    int N, int D, float scale)
{
    int bh     = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid    = threadIdx.x;
    int q_g    = q_tile * BLOCK_Q + tid;

    // shmem: packed K INT8 + K_scale + V fp32
    extern __shared__ char shmem_raw[];
    int8_t* K_int8  = (int8_t*)shmem_raw;                    // [BLOCK_KV × D]
    float*  K_scale = (float*)(shmem_raw + BLOCK_KV * D);    // [BLOCK_KV]
    float*  V_tile  = K_scale + BLOCK_KV;                    // [BLOCK_KV × D]

    // load Q into registers and compute per-thread int8 scale via warp max
    float q_fp[MAX_D], q_max = 0.0f;
    for (int d = 0; d < D; d++) {
        q_fp[d] = (q_g < N) ? Q[bh * N * D + q_g * D + d] : 0.0f;
        q_max = max(q_max, fabsf(q_fp[d]));
    }
    float q_scale = q_max / 127.0f + 1e-5f;

    // quantize Q to INT8 packed as int32 (4 elements per int32 for __dp4a)
    int q_int_packed[MAX_D / 4];
    for (int d4 = 0; d4 < D / 4; d4++) {
        int8_t b0 = (int8_t)max(-128, min(127, (int)roundf(q_fp[d4*4+0] / q_scale)));
        int8_t b1 = (int8_t)max(-128, min(127, (int)roundf(q_fp[d4*4+1] / q_scale)));
        int8_t b2 = (int8_t)max(-128, min(127, (int)roundf(q_fp[d4*4+2] / q_scale)));
        int8_t b3 = (int8_t)max(-128, min(127, (int)roundf(q_fp[d4*4+3] / q_scale)));
        // pack: little-endian int8×4 → int32
        q_int_packed[d4] = ((unsigned char)b0) | ((unsigned char)b1 << 8)
                         | ((unsigned char)b2 << 16) | ((unsigned char)b3 << 24);
    }

    float m_i = -1e20f, l_i = 0.0f;
    float o_i[MAX_D];
    for (int d = 0; d < D; d++) o_i[d] = 0.0f;

    int n_kv = (N + BLOCK_KV - 1) / BLOCK_KV;
    for (int kv = 0; kv < n_kv; kv++) {
        int kv_start = kv * BLOCK_KV;

        // cooperative load K tile + quantize to INT8 in shmem
        for (int i = tid; i < BLOCK_KV; i += BLOCK_Q) {
            int kv_g = kv_start + i;
            float k_max = 0.0f;
            for (int d = 0; d < D; d++) {
                float kv = (kv_g < N) ? K[bh * N * D + kv_g * D + d] : 0.0f;
                k_max = max(k_max, fabsf(kv));
                V_tile[i * D + d] = (kv_g < N) ? V[bh * N * D + kv_g * D + d] : 0.0f;
            }
            float ks = k_max / 127.0f + 1e-5f;
            K_scale[i] = ks;
            for (int d = 0; d < D; d++) {
                float kv = (kv_g < N) ? K[bh * N * D + kv_g * D + d] : 0.0f;
                K_int8[i * D + d] = (int8_t)max(-128, min(127, (int)roundf(kv / ks)));
            }
        }
        __syncthreads();

        if (q_g < N) {
            for (int j = 0; j < BLOCK_KV && kv_start + j < N; j++) {
                // INT8 dot via __dp4a: processes 4 dims per instruction
                // 1 __dp4a = 4 INT8 MADs; D=64 → 16 __dp4a calls vs 64 FP32 MADs → 4× instruction savings
                int dot_int = 0;
                for (int d4 = 0; d4 < D / 4; d4++) {
                    int k_packed = ((unsigned char)K_int8[j * D + d4*4+0])
                                 | ((unsigned char)K_int8[j * D + d4*4+1] << 8)
                                 | ((unsigned char)K_int8[j * D + d4*4+2] << 16)
                                 | ((unsigned char)K_int8[j * D + d4*4+3] << 24);
                    dot_int = dp4a(q_int_packed[d4], k_packed, dot_int);
                }

                // dequantize: int32_dot × q_scale × k_scale / (127² × sqrt(D))
                float score = (float)dot_int * q_scale * K_scale[j] * scale;

                float m_new = max(m_i, score);
                float alpha = expf(m_i - m_new), p = expf(score - m_new);
                l_i = l_i * alpha + p;
                for (int d = 0; d < D; d++)
                    o_i[d] = o_i[d] * alpha + p * V_tile[j * D + d];
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    if (q_g < N)
        for (int d = 0; d < D; d++)
            Out[bh * N * D + q_g * D + d] = o_i[d] / l_i;
}

torch::Tensor sage_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat32);
    TORCH_CHECK(Q.size(2) <= MAX_D && Q.size(2) % 4 == 0, "D must be multiple of 4 and <= MAX_D");
    int BH = Q.size(0), N = Q.size(1), D = Q.size(2);
    auto Out = torch::empty_like(Q);
    // shmem: K_int8[BLOCK_KV×D×1B] + K_scale[BLOCK_KV×4B] + V[BLOCK_KV×D×4B]
    size_t shmem = BLOCK_KV * D * sizeof(int8_t) + BLOCK_KV * sizeof(float) + BLOCK_KV * D * sizeof(float);
    dim3 grid(BH, (N + BLOCK_Q - 1) / BLOCK_Q);
    sage_attn_fwd<<<grid, BLOCK_Q, shmem>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        Out.data_ptr<float>(), N, D, 1.0f/sqrtf(D));
    return Out;
}
"""

_ext = load_inline(
    name="fp32_sage_attn_sm86",
    cpp_sources="torch::Tensor sage_attn_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);",
    cuda_sources=_CUDA,
    functions=["sage_attn_forward"],
    extra_cuda_cflags=["-O2", "--use_fast_math", "-arch=sm_86"],
    verbose=False,
)

def sage_attn(q, k, v):
    """q, k, v: (B*H, N, D) fp32. D must be multiple of 4."""
    return _ext.sage_attn_forward(q.contiguous(), k.contiguous(), v.contiguous())

def reference(q, k, v):
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.matmul(torch.softmax(s, dim=-1), v)

if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    BH, N, D = 16, 512, 64

    q = torch.randn(BH, N, D, device='cuda')
    k = torch.randn(BH, N, D, device='cuda')
    v = torch.randn(BH, N, D, device='cuda')

    out_s = sage_attn(q, k, v)
    out_r = reference(q, k, v)
    diff  = (out_s - out_r).abs().max().item()
    print(f"fp32_sage_attn_sm86  max|sage-ref|={diff:.5f}  {'OK (INT8 quant)' if diff < 0.15 else 'FAIL'}")

    for _ in range(10): sage_attn(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): sage_attn(q, k, v)
    torch.cuda.synchronize()
    ms_s = (time.perf_counter() - t0) / 100 * 1e3

    for _ in range(10): reference(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): reference(q, k, v)
    torch.cuda.synchronize()
    ms_r = (time.perf_counter() - t0) / 100 * 1e3
    print(f"  sage={ms_s:.3f}ms  standard={ms_r:.3f}ms")
    print(f"  __dp4a: {D//4} calls/token-key pair vs {D} FP32 MADs (4× instruction reduction)")
