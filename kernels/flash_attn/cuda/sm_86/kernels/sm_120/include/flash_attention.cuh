#pragma once

#include <torch/torch.h>
#include <algorithm>

// =============================================================================
// flash_attention.cuh — runtime kernel arguments and kernel configuration
//
// These structs are plain C++ (no template metaprogramming) and serve as the
// runtime interface between the Python caller and the CUDA kernels.
// =============================================================================

namespace flash {

// Arguments passed at kernel launch (one set per grid dispatch).
struct ForwardKernelArgs {
    using index_t = int64_t;

    void *__restrict__ Q;
    void *__restrict__ K;
    void *__restrict__ V;
    void *__restrict__ O;

    // All tensors share the same strides (row-major layout assumed).
    const index_t batch_stride;
    const index_t seq_stride;
    const index_t head_stride;

    const index_t seq_len;
    const index_t n_heads;

    const int n_Q_blocks;
    const int n_KV_blocks;
};

} // namespace flash

// ---------------------------------------------------------------------------
// FlashForwardKernelConfig — runtime description of one kernel variant.
//
// At module load time Python selects a config and we look up the matching
// pre-compiled kernel in the forward_kernels map (flash_kernels.cuh).
// ---------------------------------------------------------------------------
struct FlashForwardKernelConfig {
    const torch::ScalarType dtype;
    const int d_head;  // head dimension [64, 128]
    const int B_r;     // query tile rows [64, 128]
    const int B_c;     // key/value tile rows [32, 64, 128]
    const int n_warps; // warps per CTA [4, 8]

    const bool async_copy;
    // Pre-load next K/V block into smem as early as possible
    const bool eager_load_blocks;
    const bool swizzled;

    // Number of d_head / B_c K-fragments to load between consecutive mma calls.
    // 0 means: load the entire block into RF before the mma loop.
    const int Q_mma_load_K_fragments;
    const int K_mma_load_K_fragments;
    const int V_mma_load_K_fragments;

    // Issue the next ldmatrix before the current mma finishes (double-buffer)
    const bool mma_double_buffer_loads;
    const bool optimized_softmax;

    // Shared memory required for one CTA (Q + K + V tiles)
    int smem_bytes(int elem_size = 2) const {
        return (B_r + B_c * 2) * d_head * elem_size;
    }

    // Maximum number of resident CTAs per SM
    int num_ctas_per_sm(int max_smem_bytes) const {
        if ((n_warps == 8) || (max_smem_bytes < smem_bytes() * 2))
            return 1;
        return 2;
    }

    // Strict weak ordering so this struct can be used as a std::map key
    bool operator<(const FlashForwardKernelConfig &other) const {
        if (dtype != other.dtype)             return dtype < other.dtype;
        if (d_head != other.d_head)           return d_head < other.d_head;
        if (B_r != other.B_r)                 return B_r < other.B_r;
        if (B_c != other.B_c)                 return B_c < other.B_c;
        if (n_warps != other.n_warps)         return n_warps < other.n_warps;
        if (async_copy != other.async_copy)   return async_copy < other.async_copy;
        if (eager_load_blocks != other.eager_load_blocks)
            return eager_load_blocks < other.eager_load_blocks;
        if (swizzled != other.swizzled)       return swizzled < other.swizzled;
        if (Q_mma_load_K_fragments != other.Q_mma_load_K_fragments)
            return Q_mma_load_K_fragments < other.Q_mma_load_K_fragments;
        if (K_mma_load_K_fragments != other.K_mma_load_K_fragments)
            return K_mma_load_K_fragments < other.K_mma_load_K_fragments;
        if (V_mma_load_K_fragments != other.V_mma_load_K_fragments)
            return V_mma_load_K_fragments < other.V_mma_load_K_fragments;
        if (mma_double_buffer_loads != other.mma_double_buffer_loads)
            return mma_double_buffer_loads < other.mma_double_buffer_loads;
        if (optimized_softmax != other.optimized_softmax)
            return optimized_softmax < other.optimized_softmax;
        return false;
    }
};
