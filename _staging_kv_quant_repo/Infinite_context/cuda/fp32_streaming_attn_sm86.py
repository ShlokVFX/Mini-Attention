import math
import torch
from torch.utils.cpp_extension import load_inline

# StreamingLLM CUDA: K/V cache bounded at (sink_size + window_size) × D × 4B per head
# BLOCK_Q=32, BLOCK_KV=32: shmem = 2×32×D×4B; D=64 → 16KB → 3 blocks/SM on 48KB SM86
# per-token decode cost: O((sink+window)×D) FLOPs vs O(N×D) standard — sublinear in sequence length
# register use: q[64]+o[64]+m,l = 130 fp32 regs/thread; 32 threads × 130 = 4160 regs/block (6.3% of SM86's 65536)

_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_KV 32
#define MAX_D 128

// Grid (B, H, Q_tiles), Block BLOCK_Q threads
// Each block attends to sink_size + window_size KV positions — cache stays constant
// Shmem: K_tile[BLOCK_KV×D] + V_tile[BLOCK_KV×D] = 2×32×D×4B
__global__ void streaming_fwd(
    const float* __restrict__ Q,         // (B*H, N_q, D)
    const float* __restrict__ K_sink,    // (B*H, sink_size, D)
    const float* __restrict__ V_sink,
    const float* __restrict__ K_win,     // (B*H, window_size, D)
    const float* __restrict__ V_win,
    float*       __restrict__ Out,
    int N_q, int sink_size, int window_size, int D,
    float scale, int BLOCK_Q)
{
    int b_h   = blockIdx.x;
    int q_tile = blockIdx.y;
    int tid    = threadIdx.x;
    int q_global = q_tile * BLOCK_Q + tid;

    extern __shared__ float tiles[];
    float* K_tile = tiles;
    float* V_tile = tiles + BLOCK_KV * D;

    float q[MAX_D], o_i[MAX_D];
    float m_i = -1e20f, l_i = 0.0f;
    for (int d = 0; d < D; d++) {
        q[d]   = (q_global < N_q) ? Q[b_h * N_q * D + q_global * D + d] : 0.0f;
        o_i[d] = 0.0f;
    }

    // helper lambda-like inline for sink + window pass (identical logic, different pointer)
    // sink pass: small, usually fits in a single tile (sink_size=4 typical)
    auto process_kv = [&](const float* K_src, const float* V_src, int kv_len) {
        int n_tiles = (kv_len + BLOCK_KV - 1) / BLOCK_KV;
        for (int t = 0; t < n_tiles; t++) {
            int kv_start = t * BLOCK_KV;
            for (int i = tid; i < BLOCK_KV * D; i += BLOCK_Q) {
                int row = i / D, col = i % D;
                int g   = kv_start + row;
                K_tile[i] = (g < kv_len) ? K_src[b_h * kv_len * D + g * D + col] : 0.0f;
                V_tile[i] = (g < kv_len) ? V_src[b_h * kv_len * D + g * D + col] : 0.0f;
            }
            __syncthreads();

            if (q_global < N_q) {
                for (int j = 0; j < BLOCK_KV && kv_start + j < kv_len; j++) {
                    float score = 0.0f;
                    for (int d = 0; d < D; d++) score += q[d] * K_tile[j * D + d];
                    score *= scale;
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
    };

    // sink tokens bandwidth: sink_size × D × 8B (K+V) — typically 4×64×8=2KB per block
    process_kv(K_sink, V_sink, sink_size);
    // window tokens bandwidth: window_size × D × 8B — window=512, D=64 → 256KB per block
    process_kv(K_win,  V_win,  window_size);

    if (q_global < N_q)
        for (int d = 0; d < D; d++)
            Out[b_h * N_q * D + q_global * D + d] = o_i[d] / l_i;
}

torch::Tensor streaming_attn_forward(
    torch::Tensor Q, torch::Tensor K_sink, torch::Tensor V_sink,
    torch::Tensor K_win, torch::Tensor V_win)
{
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat32);
    int BH = Q.size(0), N_q = Q.size(1), D = Q.size(2);
    int sink_size   = K_sink.size(1);
    int window_size = K_win.size(1);
    int BLOCK_Q     = 32;

    auto Out = torch::empty_like(Q);
    size_t shmem = 2 * BLOCK_KV * D * sizeof(float);
    dim3 grid(BH, (N_q + BLOCK_Q - 1) / BLOCK_Q);

    streaming_fwd<<<grid, BLOCK_Q, shmem>>>(
        Q.data_ptr<float>(), K_sink.data_ptr<float>(), V_sink.data_ptr<float>(),
        K_win.data_ptr<float>(), V_win.data_ptr<float>(), Out.data_ptr<float>(),
        N_q, sink_size, window_size, D, 1.0f/sqrtf(D), BLOCK_Q);
    return Out;
}
"""

_ext = load_inline(
    name="fp32_streaming_attn_sm86",
    cpp_sources="""torch::Tensor streaming_attn_forward(
        torch::Tensor Q, torch::Tensor K_sink, torch::Tensor V_sink,
        torch::Tensor K_win, torch::Tensor V_win);""",
    cuda_sources=_CUDA,
    functions=["streaming_attn_forward"],
    extra_cuda_cflags=["-O2", "--use_fast_math", "-arch=sm_86"],
    verbose=False,
)


def streaming_attn(q, k, v, sink_size=4, window_size=256):
    """q,k,v: (B, H, N, D) fp32; returns (B, H, N_q, D)"""
    B, H, N_q, D = q.shape
    N_kv = k.shape[2]

    actual_sink = min(sink_size, N_kv)
    actual_win  = min(window_size, max(0, N_kv - actual_sink))

    q_3d    = q.reshape(B * H, N_q, D).contiguous()
    k_sink  = k.reshape(B * H, N_kv, D)[:, :actual_sink, :].contiguous()
    v_sink  = v.reshape(B * H, N_kv, D)[:, :actual_sink, :].contiguous()
    k_win   = k.reshape(B * H, N_kv, D)[:, N_kv - actual_win:, :].contiguous() if actual_win > 0 \
              else torch.zeros(B * H, 1, D, device=q.device)
    v_win   = v.reshape(B * H, N_kv, D)[:, N_kv - actual_win:, :].contiguous() if actual_win > 0 \
              else torch.zeros(B * H, 1, D, device=q.device)

    out = _ext.streaming_attn_forward(q_3d, k_sink, v_sink, k_win, v_win)
    return out.reshape(B, H, N_q, D)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, D = 2, 8, 512, 64
    q = torch.randn(B, H, N, D, device='cuda')
    k = torch.randn(B, H, N, D, device='cuda')
    v = torch.randn(B, H, N, D, device='cuda')

    SINK, WIN = 4, 128
    out = streaming_attn(q, k, v, sink_size=SINK, window_size=WIN)

    k_r = torch.cat([k[:, :, :SINK, :], k[:, :, N-WIN:, :]], dim=2)
    v_r = torch.cat([v[:, :, :SINK, :], v[:, :, N-WIN:, :]], dim=2)
    s   = torch.matmul(q, k_r.transpose(-2, -1)) / math.sqrt(D)
    ref = torch.matmul(torch.softmax(s, dim=-1), v_r)

    diff = (out - ref).abs().max().item()
    print(f"fp32_streaming_attn_sm86  max|cuda-ref|={diff:.6f}  {'OK' if diff < 1e-4 else 'FAIL'}")
    kv_std = N * D * 4 * 2 * B * H / 1e6
    kv_str = (SINK + WIN) * D * 4 * 2 * B * H / 1e6
    print(f"  KV memory: standard={kv_std:.1f}MB  streaming={kv_str:.1f}MB")
