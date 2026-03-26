// K8 FlashAttention — wgmma.mma_async kernel (SM90/SM120 only)
// Fixed config: Br=64, Bc=64, d_head=128, NW=4, fp16.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "include/wgmma_forward.cuh"

// ---------------------------------------------------------------------------
// Python-visible entry point
//   q, k, v, o : (B, S, H, D) contiguous fp16 tensors
//   causal      : reserved (not implemented), must be False
// ---------------------------------------------------------------------------
void forward_k8(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                torch::Tensor o, bool causal)
{
    TORCH_CHECK(!causal, "K8: causal mask not yet supported");
    TORCH_CHECK(q.scalar_type() == torch::kFloat16, "K8: fp16 only");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() &&
                v.is_contiguous() && o.is_contiguous(),
                "K8: all tensors must be contiguous");

    const int B  = q.size(0);
    const int S  = q.size(1);
    const int H  = q.size(2);
    const int D  = q.size(3);

    TORCH_CHECK(D  == flash::K8_D,  "K8: d_head must be 128");
    TORCH_CHECK(S  % flash::K8_Br == 0, "K8: seq_len must be divisible by Br=64");
    TORCH_CHECK(S  % flash::K8_Bc == 0, "K8: seq_len must be divisible by Bc=64");

    // Strides: tensor is (B, S, H, D) → head stride = D, seq stride = H*D, batch stride = S*H*D
    const int64_t seq_stride   = H * D;
    const int64_t head_stride  = D;
    const int64_t batch_stride = (int64_t)S * H * D;

    flash::ForwardKernelArgs args{
        q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
        batch_stride, seq_stride, head_stride,
        (int64_t)S, (int64_t)H,
        S / flash::K8_Br,   // n_Q_blocks
        S / flash::K8_Bc,   // n_KV_blocks
    };

    // Grid: (n_Q_blocks, H, B)
    dim3 grid(S / flash::K8_Br, H, B);
    dim3 block(flash::K8_NW * WARP_SIZE);

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaFuncSetAttribute(flash::flash_forward_k8,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         flash::K8_smem_bytes);

    flash::flash_forward_k8<<<grid, block, flash::K8_smem_bytes, stream>>>(args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_k8", &forward_k8,
          "K8 wgmma FlashAttention forward (Br=64,Bc=64,d=128,NW=4,fp16)");
}
