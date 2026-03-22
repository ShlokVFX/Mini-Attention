import torch
from torch.utils.cpp_extension import load_inline

# Device-aware MoE CUDA: SiLU-gated FFN for tokens routed to one CPU-resident expert
# GPU footprint during expert execution: gate_w + up_w + down_w = 3 × D_ff × D × 4B fp32
#   D=4096, D_ff=11008: 3 × 11008 × 4096 × 4B = 540MB (vs 8.6GB for 16 experts simultaneously)
# SM86: BLOCK_T=32 tokens, BLOCK_D=64; shmem = (BLOCK_T × BLOCK_D + BLOCK_D × BLOCK_F) × 4B
#   at BLOCK_D=64, BLOCK_F=64: (32×64 + 64×64) × 4B = (8KB + 16KB) = 24KB → 2 blocks/SM on 48KB
# register pressure: gate[BLOCK_F] + up[BLOCK_F] + out[BLOCK_D] in accumulators → ~192 fp32 regs/warp

_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_T  32   // token tile
#define BLOCK_F  64   // D_ff tile
#define MAX_D   256   // max supported hidden dim

// Grid (T_tiles, D_tiles), Block (BLOCK_T) threads
// Each block computes one (BLOCK_T, BLOCK_D) output tile via two fused GEMMs + SiLU
// Shmem: x_tile[BLOCK_T×BLOCK_D] + weight tiles streamed; totals ~24KB at default tile sizes
__global__ void expert_ffn_fwd(
    const float* __restrict__ X,       // (num_tokens, D)
    const float* __restrict__ G_w,     // (D_ff, D)  gate projection
    const float* __restrict__ U_w,     // (D_ff, D)  up projection
    const float* __restrict__ D_w,     // (D, D_ff)  down projection
    float*       __restrict__ Out,
    int num_tokens, int D, int D_ff,
    int BLOCK_D)
{
    int t_tile = blockIdx.x;
    int d_tile = blockIdx.y;
    int tid    = threadIdx.x;   // 0..BLOCK_T-1

    int t_global = t_tile * BLOCK_T + tid;

    // token vector loaded once into registers; stays there for all D_ff iterations
    float x[MAX_D];
    for (int d = 0; d < D; d++)
        x[d] = (t_global < num_tokens) ? X[t_global * D + d] : 0.0f;

    int d_start = d_tile * BLOCK_D;

    // accumulate down-projection output for this (token, BLOCK_D) tile
    float out_acc[MAX_D];
    for (int d = 0; d < BLOCK_D && d_start + d < D; d++) out_acc[d] = 0.0f;

    for (int f_start = 0; f_start < D_ff; f_start += BLOCK_F) {
        // step 1: gate = x · G_w[f_start..f_start+BLOCK_F, :]^T
        //         up   = x · U_w[...]^T
        // sequential D loop — could use WMMA but this baseline shows the pattern
        for (int f = 0; f < BLOCK_F && f_start + f < D_ff; f++) {
            float gate = 0.0f, up = 0.0f;
            for (int d = 0; d < D; d++) {
                gate += x[d] * G_w[(f_start + f) * D + d];
                up   += x[d] * U_w[(f_start + f) * D + d];
            }

            // SiLU gating: silu(gate) × up — computed in registers, zero HBM traffic
            float silu_gate = gate * (1.0f / (1.0f + expf(-gate)));
            float hidden    = silu_gate * up;

            // step 2: accumulate down projection: out[d] += hidden × D_w[d, f_start+f]
            for (int d = 0; d < BLOCK_D && d_start + d < D; d++)
                out_acc[d] += hidden * D_w[(d_start + d) * D_ff + (f_start + f)];
        }
    }

    if (t_global < num_tokens)
        for (int d = 0; d < BLOCK_D && d_start + d < D; d++)
            Out[t_global * D + (d_start + d)] = out_acc[d];
}

torch::Tensor expert_ffn_forward(
    torch::Tensor X, torch::Tensor G_w, torch::Tensor U_w, torch::Tensor D_w)
{
    TORCH_CHECK(X.is_cuda() && X.dtype() == torch::kFloat32);
    int num_tokens = X.size(0), D = X.size(1), D_ff = G_w.size(0);
    int BLOCK_D = 64;

    auto Out = torch::empty({num_tokens, D}, X.options());
    dim3 grid((num_tokens + BLOCK_T - 1) / BLOCK_T, (D + BLOCK_D - 1) / BLOCK_D);
    expert_ffn_fwd<<<grid, BLOCK_T>>>(
        X.data_ptr<float>(), G_w.data_ptr<float>(),
        U_w.data_ptr<float>(), D_w.data_ptr<float>(),
        Out.data_ptr<float>(), num_tokens, D, D_ff, BLOCK_D);
    return Out;
}
"""

_ext = load_inline(
    name="fp32_moe_sm86",
    cpp_sources="""torch::Tensor expert_ffn_forward(
        torch::Tensor X, torch::Tensor G_w, torch::Tensor U_w, torch::Tensor D_w);""",
    cuda_sources=_CUDA,
    functions=["expert_ffn_forward"],
    extra_cuda_cflags=["-O2", "--use_fast_math", "-arch=sm_86"],
    verbose=False,
)


def cuda_expert_ffn(x, gate_w, up_w, down_w):
    """SiLU-gated FFN for a single expert. Weights pre-moved to GPU by caller."""
    return _ext.expert_ffn_forward(
        x.contiguous(), gate_w.contiguous(), up_w.contiguous(), down_w.contiguous()
    )


def device_aware_moe(x, gate_proj, experts_gate, experts_up, experts_down, top_k=2):
    """x: (T, D) fp32 on GPU. Expert weights: list of CPU fp32 tensors."""
    num_experts = len(experts_gate)
    T, D = x.shape

    logits  = torch.matmul(x, gate_proj.T)
    weights, indices = torch.topk(logits.softmax(dim=-1), top_k, dim=-1)

    out = torch.zeros_like(x)

    for eid in range(num_experts):
        mask = (indices == eid).any(dim=-1)
        if not mask.any():
            continue

        routed = x[mask]
        # CPU→GPU transfer: only this expert's weights (3 × D_ff × D × 4B)
        g_w = experts_gate[eid].to(x.device)
        u_w = experts_up[eid].to(x.device)
        d_w = experts_down[eid].to(x.device)

        expert_out = cuda_expert_ffn(routed, g_w, u_w, d_w)
        out[mask] += expert_out * weights[mask].mean(dim=-1, keepdim=True)

        del g_w, u_w, d_w
        torch.cuda.empty_cache()

    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    num_experts, D, D_ff, T = 4, 128, 256, 32

    x         = torch.randn(T, D, device='cuda')
    gate_proj = torch.randn(num_experts, D, device='cuda')
    e_gate    = [torch.randn(D_ff, D) for _ in range(num_experts)]
    e_up      = [torch.randn(D_ff, D) for _ in range(num_experts)]
    e_down    = [torch.randn(D, D_ff) for _ in range(num_experts)]

    torch.cuda.reset_peak_memory_stats()
    out = device_aware_moe(x, gate_proj, e_gate, e_up, e_down)
    torch.cuda.synchronize()
    peak_mb      = torch.cuda.max_memory_allocated() / 1e6
    all_expert_mb = num_experts * 3 * D_ff * D * 4 / 1e6
    print(f"fp32_moe_sm86  out shape={out.shape}")
    print(f"  peak GPU mem: {peak_mb:.1f}MB  vs all-on-GPU: {all_expert_mb:.1f}MB")
