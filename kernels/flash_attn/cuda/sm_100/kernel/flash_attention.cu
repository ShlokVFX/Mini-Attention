#include <torch/python.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <utility>
#include <vector>

// B200 SM_100 kernel headers
#include "cuda_utils.cuh"
#include "flash_attention.cuh"
#include "flash_kernels.cuh"

using namespace flash;

// Convert the Python kernel config object into the C++ FlashForwardKernelConfig struct.
FlashForwardKernelConfig py_to_cpp_kernel_config(const py::object &py_cfg) {
    return FlashForwardKernelConfig{
        py::cast<torch::ScalarType>(
            py_cfg.attr("dtype").attr("to_torch_dtype")()),
        py::cast<int>(py_cfg.attr("d_head")),
        py::cast<int>(py_cfg.attr("B_r")),
        py::cast<int>(py_cfg.attr("B_c")),
        py::cast<int>(py_cfg.attr("n_warps")),
        py::cast<bool>(py_cfg.attr("async_copy")),
        py::cast<bool>(py_cfg.attr("eager_load_blocks")),
        py::cast<bool>(py_cfg.attr("swizzled")),
        py::cast<int>(py_cfg.attr("Q_mma_load_K_tiles")),
        py::cast<int>(py_cfg.attr("K_mma_load_K_tiles")),
        py::cast<int>(py_cfg.attr("V_mma_load_K_tiles")),
        py::cast<bool>(py_cfg.attr("mma_double_buffer_loads")),
        py::cast<bool>(py_cfg.attr("optimized_softmax"))};
}

// Shared dispatch logic — selects mma.sync or WGMMA kernel map.
decltype(auto)
flash_attention_forward(const py::object &py_cfg, const torch::Tensor &TQ,
                        const torch::Tensor &TK, const torch::Tensor &TV,
                        std::optional<at::Tensor> &out_, bool benchmark,
                        bool use_wgmma) {
    CHECK_INPUT(TQ);
    CHECK_INPUT(TK);
    CHECK_INPUT(TV);

    at::cuda::CUDAGuard device_guard{TQ.device()};
    const int compute_capability =
        cuda_device_compute_capability(TQ.device().index());
    TORCH_CHECK(compute_capability >= 100,
                "Flash Attention B200 requires SM_100 or higher (current: SM_",
                compute_capability / 10, ".", compute_capability % 10, ")");

    const auto Q_dtype = TQ.dtype();
    TORCH_CHECK(Q_dtype == torch::kFloat16 || Q_dtype == torch::kBFloat16,
                "Only fp16 and bf16 are supported");
    TORCH_CHECK(TK.dtype() == Q_dtype, "Input tensors must have the same data type");
    TORCH_CHECK(TV.dtype() == Q_dtype, "Input tensors must have the same data type");

    const FlashForwardKernelConfig cfg{py_to_cpp_kernel_config(py_cfg)};

    // Select kernel map: WGMMA (B200-optimized QK GEMM) or mma.sync baseline.
    const auto &kmap = use_wgmma ? wgmma_forward_kernels : forward_kernels;
    TORCH_CHECK(kmap.contains(cfg),
                use_wgmma
                    ? "WGMMA kernel config not found — check Bc∈{64,128}, FP16, NW%4==0"
                    : "Kernel configuration was not found in flash_kernels.cuh");
    const auto kernel = kmap.at(cfg);

    TORCH_CHECK(cfg.dtype == Q_dtype,
                "Kernel configuration dtype does not match input dtype");

    const auto batch_size = TQ.size(0);
    const auto seq_len    = TQ.size(1);
    const auto n_heads    = TQ.size(2);

    TORCH_CHECK(TQ.sizes() == TK.sizes(), "Query and key tensors have same shape");
    TORCH_CHECK(TQ.sizes() == TV.sizes(), "Query and value tensors have same shape");

    const int B_r = cfg.B_r;
    const int B_c = cfg.B_c;
    TORCH_CHECK(seq_len % B_r == 0,
                "seq_len must be a multiple of B_r");
    TORCH_CHECK(seq_len % B_c == 0,
                "seq_len must be a multiple of B_c");

    const auto batch_stride = TQ.stride(0);
    const auto seq_stride   = TQ.stride(1);
    const auto head_stride  = TQ.stride(2);

    torch::Tensor TO;
    if (out_.has_value()) {
        TO = out_.value();
        TORCH_CHECK(TO.dtype() == Q_dtype,
                    "Output tensor must have the same dtype as inputs");
        TORCH_CHECK(TQ.sizes() == TV.sizes(),
                    "Query and output tensors have same shape");
    } else {
        TO = torch::empty_like(TQ);
    }

    const int n_Q_blocks  = CEIL_DIV(seq_len, B_r);
    const int n_KV_blocks = CEIL_DIV(seq_len, B_c);
    const int n_threads   = cfg.n_warps * WARP_SIZE;

    ForwardKernelArgs args{TQ.data_ptr(), TK.data_ptr(), TV.data_ptr(),
                           TO.data_ptr(), batch_stride,  seq_stride,
                           head_stride,   seq_len,       n_heads,
                           n_Q_blocks,    n_KV_blocks};

    dim3 blockDim(n_threads);
    dim3 gridDim{static_cast<uint>(n_Q_blocks),
                 static_cast<uint>(n_heads),
                 static_cast<uint>(batch_size)};

    float runtime = 0.f;
    cudaEvent_t start, stop;

    const int smem_bytes = cfg.smem_bytes();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (benchmark) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }

    kernel<<<gridDim, blockDim, smem_bytes, stream>>>(args);

    if (benchmark) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runtime, start, stop);
    }

    return std::make_tuple(TO, runtime);
}

// ---------------------------------------------------------------------------
// flash_attention_forward_tcgen05
//
// Dispatch path for the B200-native tcgen05 kernel.
// Keyed on {dtype, head_dim, is_causal} — no tile-shape config required from
// Python; tile sizes are compile-time constants in forward_kernel_tcgen05.cuh.
// ---------------------------------------------------------------------------
decltype(auto)
flash_attention_forward_tcgen05(const torch::Tensor &TQ,
                                const torch::Tensor &TK,
                                const torch::Tensor &TV,
                                std::optional<at::Tensor> &out_,
                                bool is_causal,
                                bool benchmark) {
    CHECK_INPUT(TQ);
    CHECK_INPUT(TK);
    CHECK_INPUT(TV);

    at::cuda::CUDAGuard device_guard{TQ.device()};
    const int cc = cuda_device_compute_capability(TQ.device().index());
    TORCH_CHECK(cc >= 100,
                "tcgen05 kernel requires SM_100a (B200); found SM_",
                cc / 10, ".", cc % 10);

    const auto Q_dtype = TQ.dtype();
    TORCH_CHECK(Q_dtype == torch::kFloat16,
                "tcgen05 kernel currently supports FP16 only");
    TORCH_CHECK(TK.dtype() == Q_dtype && TV.dtype() == Q_dtype,
                "K and V must have the same dtype as Q");

    const auto batch_size = TQ.size(0);
    const auto seq_len    = TQ.size(1);
    const auto n_heads    = TQ.size(2);
    const auto head_dim   = TQ.size(3);

    TORCH_CHECK(TQ.sizes() == TK.sizes() && TQ.sizes() == TV.sizes(),
                "Q, K, V must have the same shape");
    TORCH_CHECK(seq_len % flash::TC_BLK_M == 0,
                "seq_len must be a multiple of TC_BLK_M (64)");
    TORCH_CHECK(head_dim == flash::TC_HEAD_D,
                "tcgen05 kernel requires head_dim == 64");

    const Tcgen05KernelConfig tc_cfg{Q_dtype.toScalarType(),
                                     (int)head_dim, is_causal};
    TORCH_CHECK(tcgen05_forward_kernels.contains(tc_cfg),
                "No tcgen05 kernel registered for this (dtype, head_dim, is_causal) combo");
    const auto kernel = tcgen05_forward_kernels.at(tc_cfg);

    torch::Tensor TO = out_.has_value() ? out_.value() : torch::empty_like(TQ);

    const auto batch_stride = TQ.stride(0);
    const auto seq_stride   = TQ.stride(1);
    const auto head_stride  = TQ.stride(2);

    const int n_Q_blocks  = CEIL_DIV(seq_len, flash::TC_BLK_M);
    const int n_KV_blocks = CEIL_DIV(seq_len, flash::TC_BLK_N);

    ForwardKernelArgs args{TQ.data_ptr(), TK.data_ptr(), TV.data_ptr(),
                           TO.data_ptr(), batch_stride,  seq_stride,
                           head_stride,   seq_len,       n_heads,
                           n_Q_blocks,    n_KV_blocks};

    dim3 blockDim(flash::TC_N_THREADS);
    dim3 gridDim{static_cast<uint>(n_Q_blocks),
                 static_cast<uint>(n_heads),
                 static_cast<uint>(batch_size)};

    float runtime = 0.f;
    cudaEvent_t start, stop;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (benchmark) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }

    kernel<<<gridDim, blockDim, flash::TC_SMEM_BYTES, stream>>>(args);

    if (benchmark) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&runtime, start, stop);
    }

    return std::make_tuple(TO, runtime);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward,
          py::arg("kernel_cfg"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("o"), py::arg("benchmark") = false, py::arg("use_wgmma") = false,
          "Flash Attention forward (CUDA) — B200 SM_100: mma.sync or WGMMA QK path");

    // tcgen05 forward entry point — no config object needed; tile sizes are fixed.
    m.def("forward_tcgen05", &flash_attention_forward_tcgen05,
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("o") = py::none(), py::arg("is_causal") = false,
          py::arg("benchmark") = false,
          "Flash Attention forward (tcgen05/TMEM) — B200 SM_100a, FP16, HEAD_D=64");

    // Set max dynamic smem for every kernel that needs > 48 KB (CUDA default).
    // B200 supports up to 228 KB; the attribute call unlocks the full range.
    auto set_smem = [](const auto &kmap) {
        for (const auto &[cfg, kernel] : kmap) {
            int smem_used = cfg.smem_bytes();
            if (smem_used > 48 * 1024) {
                cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_used);
            }
        }
    };
    set_smem(forward_kernels);
    set_smem(wgmma_forward_kernels);

    // tcgen05 kernel: smem = TC_SMEM_BYTES = 32 KB — fits within 48 KB default,
    // but set explicitly so future larger tiles don't silently fail.
    cudaFuncSetAttribute(
        flash_forward_kernel_tcgen05,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        flash::TC_SMEM_BYTES);
}
