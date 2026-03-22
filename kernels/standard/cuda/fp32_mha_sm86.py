import math
import torch
from torch.utils.cpp_extension import load_inline

# SM86 (RTX 3060): 128 threads = 4 warps/block
# __shfl_xor_sync latency: ~4 cycles vs ~32 cycles shared mem round-trip on SM86
# shared mem per block: (N + 32) * 4B scores + warp scratch — N=512 → 2.1KB, N=4096 → 16.5KB
# register pressure: inner loop q[d] stays in 64 regs/thread; 128 threads × 64 regs = 8192 regs/block (12.5% of SM86's 65536)
# warp occupancy: 4 warps/block; shmem = (N+32)×4B limits ~6-10 blocks/SM → 24-40 warps (37-62% of SM86's 64 max)

_CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// warp-level max via XOR butterfly: log2(32)=5 rounds, each round 1 SHFL cycle on SM86
__device__ __forceinline__ float warp_max(float v) {
    v = max(v, __shfl_xor_sync(0xffffffff, v, 16));
    v = max(v, __shfl_xor_sync(0xffffffff, v,  8));
    v = max(v, __shfl_xor_sync(0xffffffff, v,  4));
    v = max(v, __shfl_xor_sync(0xffffffff, v,  2));
    v = max(v, __shfl_xor_sync(0xffffffff, v,  1));
    return v;
}

// warp-level sum: same butterfly, 5 SHFL instructions
__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

// Grid (B, H, N): each block owns one query row — N² work distributed across B×H×N blocks
// Block: 128 threads, 4 warps; each warp handles N/4 scores in parallel
// Shmem: scores[N] fp32 + warp_buf[4] fp32 = (N+4)×4B; at N=1024 → 4.1KB per block
__global__ void mha_fwd(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ Out,
    int B, int H, int N, int D,
    float scale)
{
    int b = blockIdx.x, h = blockIdx.y, q_pos = blockIdx.z;
    int tid = threadIdx.x, nthreads = blockDim.x;
    int num_warps = nthreads / 32, warp_id = tid / 32, lane_id = tid % 32;

    extern __shared__ float shmem[];
    float* scores   = shmem;
    float* warp_buf = shmem + N;   // 4 floats, reused for max then sum

    int bh_off = (b * H + h) * N * D;
    const float* q_row = Q + bh_off + q_pos * D;
    const float* K_bh  = K + bh_off;
    const float* V_bh  = V + bh_off;

    // step 1: dot products — inner D-loop stays in registers (no shmem traffic)
    for (int k = tid; k < N; k += nthreads) {
        float dot = 0.0f;
        for (int d = 0; d < D; d++) dot += q_row[d] * K_bh[k * D + d];
        scores[k] = dot * scale;
    }
    __syncthreads();

    // step 2a: find global max — warp reduce + cross-warp via shmem (one __syncthreads)
    float lmax = -1e20f;
    for (int k = tid; k < N; k += nthreads) lmax = max(lmax, scores[k]);
    lmax = warp_max(lmax);
    if (lane_id == 0) warp_buf[warp_id] = lmax;
    __syncthreads();
    if (tid == 0) {
        float gmax = -1e20f;
        for (int w = 0; w < num_warps; w++) gmax = max(gmax, warp_buf[w]);
        warp_buf[0] = gmax;
    }
    __syncthreads();
    float gmax = warp_buf[0];

    // step 2b: exp + sum
    float lsum = 0.0f;
    for (int k = tid; k < N; k += nthreads) { scores[k] = expf(scores[k] - gmax); lsum += scores[k]; }
    lsum = warp_sum(lsum);
    if (lane_id == 0) warp_buf[warp_id] = lsum;
    __syncthreads();
    if (tid == 0) {
        float gs = 0.0f;
        for (int w = 0; w < num_warps; w++) gs += warp_buf[w];
        warp_buf[0] = gs;
    }
    __syncthreads();
    float gsum = warp_buf[0];

    for (int k = tid; k < N; k += nthreads) scores[k] /= gsum;
    __syncthreads();

    // step 3: weighted V sum — each thread owns D/nthreads output elements
    // inner k-loop reads scores[] from shmem: N × 4B read per output element
    float* out_row = Out + bh_off + q_pos * D;
    for (int d = tid; d < D; d += nthreads) {
        float val = 0.0f;
        for (int k = 0; k < N; k++) val += scores[k] * V_bh[k * D + d];
        out_row[d] = val;
    }
}

torch::Tensor mha_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
{
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat32);
    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    auto Out = torch::empty_like(Q);
    // 128 threads: 4 warps at 32 threads each; shmem = (N+32)*4B
    mha_fwd<<<dim3(B,H,N), 128, (N+32)*sizeof(float)>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        Out.data_ptr<float>(), B, H, N, D, 1.0f/sqrtf(D));
    return Out;
}
"""

_ext = load_inline(
    name="fp32_mha_sm86",
    cpp_sources="torch::Tensor mha_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);",
    cuda_sources=_CUDA_SRC,
    functions=["mha_forward"],
    extra_cuda_cflags=["-O2", "--use_fast_math", "-arch=sm_86"],
    verbose=False,
)

def cuda_mha(q, k, v):
    return _ext.mha_forward(q.contiguous(), k.contiguous(), v.contiguous())

def reference(q, k, v):
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.matmul(torch.softmax(s, dim=-1), v)

if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 512, 64
    q = torch.randn(B, H, N, D, device='cuda')
    k = torch.randn(B, H, N, D, device='cuda')
    v = torch.randn(B, H, N, D, device='cuda')

    diff = (cuda_mha(q, k, v) - reference(q, k, v)).abs().max().item()
    print(f"fp32_mha_sm86  max|cuda-ref|={diff:.6f}  {'OK' if diff < 1e-4 else 'FAIL'}")

    for _ in range(10): cuda_mha(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): cuda_mha(q, k, v)
    torch.cuda.synchronize()
    print(f"  {(time.perf_counter()-t0)/100*1e3:.3f} ms/call")
