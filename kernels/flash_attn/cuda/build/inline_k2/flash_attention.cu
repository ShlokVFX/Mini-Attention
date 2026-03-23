#include <torch/python.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <utility>
#include <vector>
// >>> cuda_utils.cuh

namespace flash {

#define CHECK_CUDA(x)                                                          \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

#ifndef CUDA_CHECK_AND_EXIT
#define CUDA_CHECK_AND_EXIT(error)                                             \
    {                                                                          \
        auto status = static_cast<cudaError_t>(error);                         \
        if (status != cudaSuccess) {                                           \
            std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":"  \
                      << __LINE__ << std::endl;                                \
            std::exit(status);                                                 \
        }                                                                      \
    }
#endif

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__device__ __forceinline__ bool is_cta_leader() { return threadIdx.x == 0; }

inline int cuda_device_num_sms(int device) {
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    return sms;
}

inline int cuda_device_max_smem_bytes(int device) {
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);
    return max_smem;
}

inline int cuda_device_compute_capability(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.major * 10 + prop.minor;
}

} // namespace flash
// <<< cuda_utils.cuh
// >>> flash_attention.cuh

#include <torch/torch.h>
#include <algorithm>

namespace flash {

struct ForwardKernelArgs {
    using index_t = int64_t;

    void *__restrict__ Q;
    void *__restrict__ K;
    void *__restrict__ V;
    void *__restrict__ O;

    // We assume all strides are the same across all inputs, and that
    // the tensors are all row major.
    const index_t batch_stride;
    const index_t seq_stride;
    const index_t head_stride;

    const index_t seq_len;
    const index_t n_heads;

    const int n_Q_blocks;
    const int n_KV_blocks;
};
} // namespace flash

// FlashForwardKernelConfig contains the configuration for a kernel.
// For choosing the kernel configuration at runtime, We use a map of kernel
// configs to kernels. The official repo uses static switches, which is cleaner
// and faster.
struct FlashForwardKernelConfig {
    const torch::ScalarType dtype;
    const int d_head;  // [64, 128]
    const int B_r;     // [64, 128]
    const int B_c;     // [32, 64, 128]
    const int n_warps; // [4, 8]. 8 only when B_r = 128

    const bool async_copy;
    // If true, load K and V block tiles into smem as soon as we can.
    const bool eager_load_blocks;
    const bool swizzled;

    const int Q_mma_load_K_fragments;
    const int K_mma_load_K_fragments;
    const int V_mma_load_K_fragments;

    // if true, call ldmatrix for the next iter before calling mma.
    const bool mma_double_buffer_loads;
    const bool optimized_softmax;

    int smem_bytes(int elem_size = 2) const {
        return (B_r + B_c * 2) * d_head * elem_size;
    }

    int num_ctas_per_sm(int max_smem_bytes) const {
        // The max # ctas will be 2 or less due to register limits.
        if ((n_warps == 8) || (max_smem_bytes < smem_bytes() * 2)) {
            return 1;
        }

        return 2;
    }

    bool operator<(const FlashForwardKernelConfig &other) const {
        if (dtype != other.dtype) {
            return dtype < other.dtype;
        }
        if (d_head != other.d_head) {
            return d_head < other.d_head;
        }
        if (B_r != other.B_r) {
            return B_r < other.B_r;
        }
        if (B_c != other.B_c) {
            return B_c < other.B_c;
        }
        if (n_warps != other.n_warps) {
            return n_warps < other.n_warps;
        }
        if (async_copy != other.async_copy) {
            return async_copy < other.async_copy;
        }
        if (eager_load_blocks != other.eager_load_blocks) {
            return eager_load_blocks < other.eager_load_blocks;
        }
        if (swizzled != other.swizzled) {
            return swizzled < other.swizzled;
        }
        if (Q_mma_load_K_fragments != other.Q_mma_load_K_fragments) {
            return Q_mma_load_K_fragments < other.Q_mma_load_K_fragments;
        }
        if (K_mma_load_K_fragments != other.K_mma_load_K_fragments) {
            return K_mma_load_K_fragments < other.K_mma_load_K_fragments;
        }
        if (V_mma_load_K_fragments != other.V_mma_load_K_fragments) {
            return V_mma_load_K_fragments < other.V_mma_load_K_fragments;
        }
        if (mma_double_buffer_loads != other.mma_double_buffer_loads) {
            return mma_double_buffer_loads < other.mma_double_buffer_loads;
        }
        if (optimized_softmax != other.optimized_softmax) {
            return optimized_softmax < other.optimized_softmax;
        }
        return false; // Equal configurations
    }
};
// <<< flash_attention.cuh
// >>> flash_kernels.cuh
// This file is auto-generated in "gen_kernel_instantiations.py".


#include <map>

// >>> flash_attention.cuh
// <<< flash_attention.cuh
// >>> forward_kernel.cuh

#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// >>> common.h

#define FA_UNROLL _Pragma("unroll")
#define FA_DEVICE __forceinline__ __device__
#define FA_DEVICE_CONSTEXPR __forceinline__ __device__ constexpr

#define WARP_SIZE 32
#define SHFL_ENTIRE_WARP_MASK 0xffffffff

#define B16_BYTES 2
#define BYTES_PER_VEC4_ACCESS 16
#define ELEMS_PER_VEC4_ACCESS (BYTES_PER_VEC4_ACCESS / B16_BYTES)

// mma/ldmatrix related constants
#define MMA_A_REGS_PER_ROW 2
#define MMA_A_REGS_PER_COL 2
#define MMA_B_REGS_PER_ROW 2
#define MMA_B_REGS_PER_COL 1
#define MMA_C_REGS_PER_ROW 1
#define MMA_C_REGS_PER_COL 2

#define N_REGS_PER_F32_ACCUM_FRAGMENT 2

#define LDMATRIX_MAT_SIZE 8
#define ROWS_PER_FRAGMENT LDMATRIX_MAT_SIZE
#define COLS_PER_FRAGMENT LDMATRIX_MAT_SIZE

#define GSM_LDST_ROWS_PER_ITER 4

#define N_BUFFER_STAGES 2
// <<< common.h
// >>> flash_attention.cuh
// <<< flash_attention.cuh
// >>> gemm.cuh

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

// >>> common.h
// <<< common.h
// >>> ptx_functions.cuh

#include <cuda_bf16.h>
#include <cuda_fp16.h>
// >>> common.h
// <<< common.h

namespace flash {

template <bool async>
FA_DEVICE void cp_async_commit() {
    if constexpr (async) {
        asm volatile("cp.async.commit_group;");
    }
}

template <int ngroups, bool async>
FA_DEVICE void cp_async_wait() {
    if constexpr (async) {
        asm volatile("cp.async.wait_group %0;" ::"n"(ngroups));
    }
}

template <bool async>
FA_DEVICE_CONSTEXPR void cp_async_commit_and_wait_all() {
    if constexpr (async) {
        cp_async_commit<async>();
        cp_async_wait<0, async>();
    }
}

template <int size, typename T>
FA_DEVICE void cp_async(T *smem_to, T *gmem_from) {
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_to);
    // The .cg option bypasses the L1 cache.
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
                 :
                 : "r"(smem_ptr), "l"(gmem_from), "n"(size));
}

template <typename T>
FA_DEVICE void ldmatrix_x4(T *load_from, uint32_t &a1, uint32_t &a2,
                           uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                 "{%0, %1, %2, %3}, [%4];"
                 : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                 : "r"(smem_ptr));
}

template <typename T>
FA_DEVICE void ldmatrix_x4_transpose(T *load_from, uint32_t &a1, uint32_t &a2,
                                     uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16"
                 "{%0, %1, %2, %3}, [%4];"
                 : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                 : "r"(smem_ptr));
}

template <typename value_t>
FA_DEVICE void
mma_m16n8k16_f32_accum(float &d1, float &d2, float &d3, float &d4,
                       uint32_t const &a1, uint32_t const &a2,
                       uint32_t const &a3, uint32_t const &a4,
                       uint32_t const &b1, uint32_t const &b2, float const &c1,
                       float const &c2, float const &c3, float const &c4) {
    static_assert(std::is_same_v<value_t, half> ||
                      std::is_same_v<value_t, nv_bfloat16>,
                  "value_t must be either half or nv_bfloat16");

    if constexpr (std::is_same_v<value_t, half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     " { %0, %1, %2, %3 }, "
                     " { %4, %5, %6, %7 }, "
                     " { %8, %9 }, "
                     " { %10, %11, %12, %13 }; "
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     " { %0, %1, %2, %3 }, "
                     " { %4, %5, %6, %7 }, "
                     " { %8, %9 }, "
                     " { %10, %11, %12, %13 }; "
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    }
}

} // namespace flash// <<< ptx_functions.cuh
// >>> utils.h

namespace flash {

constexpr int constexpr_min(int a, int b) { return (a < b) ? a : b; }

constexpr int constexpr_log2_floor(int n) { return std::__bit_width(n) - 1; }

} // namespace flash
// <<< utils.h

namespace flash {

// Dimensions of the mma instruction we're using
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define MMA_M_FRAGMENTS_PER_ITER 2 // (MMA_M / LDMATRIX_MAT_SIZE)
#define MMA_N_FRAGMENTS_PER_ITER 1 // (MMA_N / LDMATRIX_MAT_SIZE)
#define MMA_K_FRAGMENTS_PER_ITER 2 // (MMA_K / LDMATRIX_MAT_SIZE)

template <typename _A_t, typename _B_t, typename _C_t, int total_K_fragments,
          int load_K_fragments_per_iter, typename value_t_>
struct GEMM {
    using A_t = _A_t;
    using B_t = _B_t;
    using C_t = _C_t;
    using value_t = value_t_;

    static constexpr int TotalKTiles = total_K_fragments;
    static constexpr int LoadKTilesPerIter = load_K_fragments_per_iter;

    static constexpr bool DoubleBufferA =
        !A_t::load_entire_block_into_rf && A_t::mma_load_stages > 1;
    static constexpr bool DoubleBufferB =
        !B_t::load_entire_block_into_rf && B_t::mma_load_stages > 1;
    static constexpr bool DoubleBuffer = DoubleBufferA || DoubleBufferB;
};

// warp_fragment_mma_f32_accum
template <typename value_t, const int M_fragments, const int N_fragments,
          const int K_fragments_A, const int K_fragments_B,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][K_fragments_A],
    uint32_t (&regs_B)[N_fragments][K_fragments_B],
    accum_t (&regs_C)[M_fragments][N_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT],
    int A_col_fragment_offset = 0, int B_col_fragment_offset = 0) {
    constexpr int K_iters = constexpr_min(K_fragments_A, K_fragments_B);
    FA_UNROLL
    for (int k = 0; k < K_iters; k += MMA_K_FRAGMENTS_PER_ITER) {
        FA_UNROLL
        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
            FA_UNROLL
            for (int n = 0; n < N_fragments; n += MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_f32_accum<value_t>(
                    regs_C[m][n * 2], regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2], regs_C[m + 1][n * 2 + 1],
                    regs_A[m][k + A_col_fragment_offset],
                    regs_A[m + 1][k + A_col_fragment_offset],
                    regs_A[m][k + 1 + A_col_fragment_offset],
                    regs_A[m + 1][k + 1 + A_col_fragment_offset],
                    regs_B[n][k + B_col_fragment_offset],
                    regs_B[n][k + 1 + B_col_fragment_offset], regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1], regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1]);
            }
        }
    }
}

template <typename GEMM>
FA_DEVICE_CONSTEXPR void matmul(typename GEMM::A_t &A, typename GEMM::B_t &B,
                                typename GEMM::C_t &C) {
    using A_t = typename GEMM::A_t;
    using B_t = typename GEMM::B_t;
    using value_t = typename GEMM::value_t;

    constexpr int A_stage_toggle = A_t::mma_load_stages - 1;
    constexpr int B_stage_toggle = B_t::mma_load_stages - 1;

    int A_stage = 0;
    int B_stage = 0;

    if constexpr (GEMM::DoubleBufferA) {
        A.copy_SM2RF(A_stage);
    }
    if constexpr (GEMM::DoubleBufferB) {
        B.copy_SM2RF(B_stage);
    }

    FA_UNROLL
    for (int k_outer_fragment = 0; k_outer_fragment < GEMM::TotalKTiles;
         k_outer_fragment += GEMM::LoadKTilesPerIter) {
        if constexpr (!A_t::load_entire_block_into_rf ||
                      !B_t::load_entire_block_into_rf) {
            int k_load_fragment =
                k_outer_fragment +
                (GEMM::DoubleBuffer ? GEMM::LoadKTilesPerIter : 0);
            if (k_load_fragment < GEMM::TotalKTiles) {
                if constexpr (!A_t::load_entire_block_into_rf) {
                    A.copy_SM2RF(A_stage_toggle ^ A_stage, k_load_fragment);
                }
                if constexpr (!B_t::load_entire_block_into_rf) {
                    B.copy_SM2RF(B_stage_toggle ^ B_stage, k_load_fragment);
                }
            }
        }

        // Perform tile-wise outer products.
        int A_col_offset =
            A_t::load_entire_block_into_rf ? k_outer_fragment : 0;
        int B_col_offset =
            B_t::load_entire_block_into_rf ? k_outer_fragment : 0;
        warp_fragment_mma_f32_accum<value_t>(A.data(A_stage), B.data(B_stage),
                                             C.data(), A_col_offset,
                                             B_col_offset);

        A_stage ^= A_stage_toggle;
        B_stage ^= B_stage_toggle;
    }
}

} // namespace flash
// <<< gemm.cuh
// >>> ptx_functions.cuh
// <<< ptx_functions.cuh
// >>> softmax.cuh

// >>> common.h
// <<< common.h

namespace flash {

/*
Each group of 4 threads contains a row.

*/

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
scale_S_accum(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
              const accum_t &softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        FA_UNROLL
        for (int k = 0; k < KV_accum_fragments; ++k) {
            S_accum[q][k] *= softmax_scale;
        }
    }
}

template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
calc_row_max(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
             accum_t (&m_next)[QO_fragments], accum_t (&m_cur)[QO_fragments]) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        m_next[q] = m_cur[q];

        // Calculate max for row across all in-thread registers.
        FA_UNROLL
        for (int k = 0; k < KV_accum_fragments; ++k) {
            m_next[q] = max(m_next[q], S_accum[q][k]);
        }

        // Group reduction
        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 2),
                        m_next[q]);
        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 1),
                        m_next[q]);
    }
}

template <bool optimized_softmax, int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
scale_l_O(accum_t (&m_next)[QO_fragments], accum_t (&m_cur)[QO_fragments],
          accum_t (&l)[QO_fragments],
          accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
          accum_t softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        accum_t scale;
        if constexpr (optimized_softmax) {
            scale = exp2f((m_cur[q] - m_next[q]) * softmax_scale);
        } else {
            scale = expf(m_cur[q] - m_next[q]);
        }
        m_cur[q] = m_next[q];
        l[q] *= scale;
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] *= scale;
        }
    }
}

template <bool optimized_softmax, int QO_fragments, int KV_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
exponentiate_tensor(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
                    accum_t (&m)[QO_fragments], accum_t softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        accum_t max_scaled;
        if constexpr (optimized_softmax) {
            max_scaled = m[q] * softmax_scale;
        }
        FA_UNROLL
        for (int k = 0; k < KV_accum_fragments; ++k) {
            if constexpr (optimized_softmax) {
                S_accum[q][k] =
                    exp2f(S_accum[q][k] * softmax_scale - max_scaled);
            } else {
                S_accum[q][k] = expf(S_accum[q][k] - m[q]);
            }
        }
    }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
update_row_exp_sum(accum_t (&P_accum)[QO_fragments][d_head_accum_fragments],
                   accum_t (&l)[QO_fragments]) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        FA_UNROLL
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            l[q] += P_accum[q][d_head];
        }
    }
}

template <int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void final_softmax_normalization(
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]) {
    // Finish summing row_sums across all threads in the same row.
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
    }

    // Final row-wise O softmax normalization.
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        FA_UNROLL
        for (int d_head = 0; d_head < d_head_accum_fragments; ++d_head) {
            O_accum[q][d_head] /= l[q];
        }
    }
}

} // namespace flash// <<< softmax.cuh
// >>> static_kernel_configuration.cuh

// >>> common.h
// <<< common.h
// >>> flash_attention.cuh
// <<< flash_attention.cuh
// >>> gemm.cuh
// <<< gemm.cuh
// >>> load_store.cuh

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// >>> common.h
// <<< common.h
// >>> ptx_functions.cuh
// <<< ptx_functions.cuh
// >>> swizzling.cuh

// >>> common.h
// <<< common.h
// >>> utils.h
// <<< utils.h

namespace flash {

template <int col_fragments>
FA_DEVICE_CONSTEXPR int swizzled_col_fragment(int row, int col_fragment) {
    static_assert(col_fragments % ELEMS_PER_VEC4_ACCESS == 0,
                  "# col tiles is a multiple of # elems");

    // The % ELEMS_PER_VEC4_ACCESS makes sure that the swizzled column stays
    // within the same 8 element window.
    return (row % ELEMS_PER_VEC4_ACCESS) ^ col_fragment;
}

template <int col_fragments, bool swizzle>
FA_DEVICE_CONSTEXPR int get_smem_col_fragment(const int row,
                                              const int col_fragment) {
    return swizzle ? swizzled_col_fragment<col_fragments>(row, col_fragment)
                   : col_fragment;
}

template <const int col_fragments, const bool swizzled>
FA_DEVICE_CONSTEXPR int get_smem_offset(const int row, const int col) {
    const int offset = row * col_fragments + col;
    if constexpr (swizzled) {
        return swizzle_cute<col_fragments>(offset);
    } else {
        return offset;
    }
}

} // namespace flash
// <<< swizzling.cuh

namespace flash {

struct LDSTCommon {
    const bool swizzled;
    const bool async_copy;
};

struct TileLayout {
    const int row_fragments;
    const int col_fragments;
};

// constexpr non-type template parameter containing parameters for LDST for a
// block (Q, K, V, or O) from gmem to smem and vice versa, and also loading from
// smem to the RF.
struct TensorLDSTConfig {
    // This contains the # of (8, 8) tiles that each warp will load/store
    // between gmem and smem.
    const TileLayout GSM;
    // contains the # of fragments that each warp will compute on.
    const TileLayout RF;
    const LDSTCommon Common;
    const bool transposed;
    const int block_size;
    const int smem_cols;

    // this is the # of rows a warp in a thread-block independently
    // loads/stores. it is equivalent to GSM.row_fragments * 8.
    const int warp_ldst_rows;

    // Whether not the warp will compute over the entire block.
    // This is false for (Q&O&S) and true for (K&V).
    const bool compute_over_entire_block;

    const bool load_entire_block_into_rf;
    const int mma_load_stages;
};

template <typename T>
struct GM2SM_async {
    FA_DEVICE_CONSTEXPR void operator()(T *gmem, T *smem) {
        cp_async<BYTES_PER_VEC4_ACCESS>(smem, gmem);
    }
};

template <typename T>
struct GM2SM {
    FA_DEVICE_CONSTEXPR void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(smem)[0] = reinterpret_cast<uint4 *>(gmem)[0];
    }
};

template <typename T>
struct SM2GM {
    FA_DEVICE_CONSTEXPR void operator()(T *gmem, T *smem) {
        reinterpret_cast<uint4 *>(gmem)[0] = reinterpret_cast<uint4 *>(smem)[0];
    }
};

// Copy a (B_r, d_head) or (B_c, d_head) block from gmem to smem or vice
// versa. Each warp independently loads a (seq_len_per_warp, d_head) block.
// Each inner iteration loads a (4, 64) tile, where each row is loaded by a
// group of 8 consecutive threads.
// In the edge case that we're loading a (128, 64) block with 8 warps, each warp
template <typename op, /* either GM2SM_async or SM2GM */
          TensorLDSTConfig CFG, typename value_t, typename index_t = int64_t>
FA_DEVICE_CONSTEXPR void copy_block_GSM(value_t *gmem, value_t *smem,
                                        index_t gmem_seq_stride,
                                        const int lane_id) {
    constexpr int n_row_iters =
        CFG.GSM.row_fragments * ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER;

    constexpr int col_fragments_per_iter = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;
    constexpr int col_fragments_per_row = CFG.smem_cols / COLS_PER_FRAGMENT;

    const int thread_row = lane_id / col_fragments_per_iter;
    const int thread_col_fragment = lane_id % col_fragments_per_iter;

    FA_UNROLL
    for (int r = 0; r < n_row_iters; ++r) {
        const int cur_row = r * GSM_LDST_ROWS_PER_ITER + thread_row;
        FA_UNROLL
        for (int c = 0; c < col_fragments_per_row;
             c += col_fragments_per_iter) {
            const int gmem_col_fragment = c + thread_col_fragment;
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments_per_row,
                                      CFG.Common.swizzled>(cur_row,
                                                           gmem_col_fragment);

            op()(&gmem[cur_row * gmem_seq_stride +
                       gmem_col_fragment * COLS_PER_FRAGMENT],
                 &smem[cur_row * CFG.smem_cols +
                       smem_col_fragment * COLS_PER_FRAGMENT]);
        }
    }
}

// Loads matrix fragments in smem into registers.
// Each ldmatrix.x4 instruction loads a (16, 16) chunk, i.e. (2, 2) fragments.
// For this non-transposed version, the shape of the smem matches rmem, i.e.
// shape(rmem) = (r_r, r_c) = (s_r / 8, s_c / 8).
// This will be used to copy Q and K.
template <TensorLDSTConfig CFG, typename value_t>
FA_DEVICE_CONSTEXPR void copy_warp_fragment_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments], value_t *smem,
    const int lane_id, const int col_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;

    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

    FA_UNROLL
    for (int r = 0; r < CFG.RF.row_fragments; r += row_fragments_per_iter) {
        const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;
        FA_UNROLL
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(
                    cur_row, thread_col_fragment + c + col_fragment_offset);

            ldmatrix_x4(&smem[cur_row * CFG.smem_cols +
                              smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                        regs[r][c], regs[r + 1][c], regs[r][c + 1],
                        regs[r + 1][c + 1]);
        }
    }
}

// Loads matrix fragments in smem into registers.
// Each ldmatrix.x4 instruction loads a (16, 16) chunk, i.e. (2, 2) fragments.
// For this transposed version, the shape of the smem matches the transpose of
// rmem, i.e. shape(rmem) = (r_r, r_c) = (s_c / 8, s_r / 8).
// This will be used to copy V.
template <TensorLDSTConfig CFG, typename value_t>
FA_DEVICE_CONSTEXPR void copy_warp_fragment_transposed_SM2RF(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments], value_t *smem,
    const int lane_id, const int row_fragment_offset = 0) {
    constexpr int row_fragments_per_iter = 2;
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;

    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;
    constexpr int col_fragments_per_iter = WARP_SIZE / rows_per_iter;

    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id / rows_per_iter;

    FA_UNROLL
    for (int r = 0; r < CFG.RF.col_fragments; r += row_fragments_per_iter) {
        const int cur_row =
            thread_row + (r + row_fragment_offset) * ROWS_PER_FRAGMENT;
        FA_UNROLL
        for (int c = 0; c < CFG.RF.row_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(
                    cur_row, thread_col_fragment + c);

            ldmatrix_x4_transpose(
                &smem[cur_row * CFG.smem_cols +
                      smem_col_fragment * ELEMS_PER_VEC4_ACCESS],
                regs[c][r], regs[c][r + 1], regs[c + 1][r], regs[c + 1][r + 1]);
        }
    }
}

// Copies matrix fragments in rmem to smem.
// Each iteration of the inner loop copies a (8, 8) tile, i.e. a single
// fragment. This will be used to copy O.
template <TensorLDSTConfig CFG, typename value_t>
FA_DEVICE_CONSTEXPR void copy_warp_fragment_RF2SM(
    uint32_t (&regs)[CFG.RF.row_fragments][CFG.RF.col_fragments], value_t *smem,
    const int lane_id) {
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT;
    constexpr int col_fragments_per_iter = 1;
    constexpr int col_fragments = CFG.smem_cols / ELEMS_PER_VEC4_ACCESS;

    constexpr int elems_per_store = 2;
    const int thread_row = lane_id / 4;
    const int thread_inner_col = (lane_id % 4) * elems_per_store;

    FA_UNROLL
    for (int r = 0; r < CFG.RF.row_fragments; ++r) {
        const int cur_row = thread_row + r * rows_per_iter;
        FA_UNROLL
        for (int c = 0; c < CFG.RF.col_fragments; c += col_fragments_per_iter) {
            const int smem_col_fragment =
                get_smem_col_fragment<col_fragments, CFG.Common.swizzled>(
                    cur_row, c);

            reinterpret_cast<uint32_t *>(
                &smem[cur_row * CFG.smem_cols +
                      (smem_col_fragment * ELEMS_PER_VEC4_ACCESS +
                       thread_inner_col)])[0] = regs[r][c];
        }
    }
}

template <typename value_t, int M_fragments, int N_fragments>
FA_DEVICE_CONSTEXPR void
convert_to_16_bit_dtype(float (&src_float)[M_fragments][N_fragments * 2],
                        uint32_t (&dest_uint)[M_fragments][N_fragments]) {
    using value2_t =
        std::conditional_t<std::is_same_v<value_t, half>, half2, nv_bfloat162>;

    float2(&src)[M_fragments][N_fragments] =
        reinterpret_cast<float2(&)[M_fragments][N_fragments]>(src_float);
    value2_t(&dest)[M_fragments][N_fragments] =
        reinterpret_cast<value2_t(&)[M_fragments][N_fragments]>(dest_uint);
    FA_UNROLL
    for (int m = 0; m < M_fragments; ++m) {
        FA_UNROLL
        for (int n = 0; n < N_fragments; ++n) {
            if constexpr (std::is_same_v<value_t, half>) {
                dest[m][n] = __float22half2_rn(src[m][n]);
            } else {
                dest[m][n] = __float22bfloat162_rn(src[m][n]);
            }
        }
    }
}

} // namespace flash
// <<< load_store.cuh
// >>> tensor.cuh

// >>> common.h
// <<< common.h
// >>> load_store.cuh
// <<< load_store.cuh

namespace flash {

template <typename value_t, int N>
struct RFVector {
    static constexpr int size = N;
    value_t regs[N];

    FA_DEVICE_CONSTEXPR value_t &operator[](int idx) { return regs[idx]; }
};

template <typename value_t, int n_copies, int row_fragments, int col_fragments>
struct RFMatrix {
    using storage_t = std::conditional_t<sizeof(value_t) == 4, float, uint32_t>;
    static constexpr int regs_per_fragment = sizeof(value_t) / 2;
    static constexpr int rows = row_fragments;
    static constexpr int cols = col_fragments * regs_per_fragment;

    storage_t regs[n_copies][rows][cols];

    FA_DEVICE_CONSTEXPR storage_t (&data(const int stage = 0))[rows][cols] {
        return reinterpret_cast<storage_t(&)[rows][cols]>(regs[stage]);
    }

    FA_DEVICE_CONSTEXPR void zero() {
        FA_UNROLL
        for (int i = 0; i < n_copies; ++i) {
            FA_UNROLL
            for (int j = 0; j < rows; ++j) {
                FA_UNROLL
                for (int k = 0; k < cols; ++k) {
                    regs[i][j][k] = 0;
                }
            }
        }
    }
};

// MatrixLDST is an object that provides ldst and conversion functionality for a
// block in memory. The scope of the object involves all the levels of memory
// (gmem, smem, and the rf). Admittedly, this class does too much, but I didn't
// want to overengineer it given the scope of this project.
template <TensorLDSTConfig ldst, typename value_t, typename index_t = int64_t>
struct MatrixLDST {
    // Static configuration
    using matrix_storage_t =
        RFMatrix<value_t, ldst.mma_load_stages, ldst.RF.row_fragments,
                 ldst.RF.col_fragments>;
    using GM2SM_op = std::conditional_t<ldst.Common.async_copy,
                                        GM2SM_async<value_t>, GM2SM<value_t>>;

    using SM2GM_op = SM2GM<value_t>;
    static constexpr int mma_load_stages = ldst.mma_load_stages;
    static constexpr bool load_entire_block_into_rf =
        ldst.load_entire_block_into_rf;
    static constexpr bool transposed = ldst.transposed;

    // Runtime properties
    value_t *gmem_ptr;
    index_t gmem_seq_stride;
    // The location in memory used to load fragments from smem to rmem.
    value_t *smem_srm_ptr;
    // The location in memory that the warp writes to for Q, K, V from gmem to
    // smem and O for smem to gmem.
    value_t *smem_gsm_ptr;

    const int lane_id;

    matrix_storage_t storage;

    FA_DEVICE MatrixLDST(value_t *gmem_block_ptr, index_t _gmem_seq_stride,
                         value_t *_smem_ptr)
        : lane_id(threadIdx.x % WARP_SIZE) {
        const int warp_rank = threadIdx.x / WARP_SIZE;

        const index_t warp_seq = ldst.warp_ldst_rows * warp_rank;

        gmem_seq_stride = _gmem_seq_stride;
        gmem_ptr = gmem_block_ptr + warp_seq * gmem_seq_stride;

        smem_gsm_ptr = _smem_ptr + warp_seq * ldst.smem_cols;
        smem_srm_ptr =
            ldst.compute_over_entire_block ? _smem_ptr : smem_gsm_ptr;
    }

    FA_DEVICE_CONSTEXPR void zero() { storage.zero(); }

    FA_DEVICE_CONSTEXPR typename matrix_storage_t::storage_t (&data(
        const int stage = 0))[matrix_storage_t::rows][matrix_storage_t::cols] {
        return storage.data(stage);
    }

    FA_DEVICE_CONSTEXPR void advance_gmem_block() {
        gmem_ptr += ldst.block_size * gmem_seq_stride;
    }

    FA_DEVICE_CONSTEXPR void copy_GM2SM() {
        copy_block_GSM<GM2SM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride,
                                       lane_id);
    }

    FA_DEVICE_CONSTEXPR void copy_SM2GM() {
        copy_block_GSM<SM2GM_op, ldst>(gmem_ptr, smem_gsm_ptr, gmem_seq_stride,
                                       lane_id);
    }

    FA_DEVICE_CONSTEXPR void copy_SM2RF(int stage = 0, int tile_offset = 0) {
        if constexpr (!transposed) {
            copy_warp_fragment_SM2RF<ldst, value_t>(
                storage.data(stage), smem_srm_ptr, lane_id, tile_offset);
        } else {
            copy_warp_fragment_transposed_SM2RF<ldst, value_t>(
                storage.data(stage), smem_srm_ptr, lane_id, tile_offset);
        }
    }

    FA_DEVICE_CONSTEXPR void copy_RF2SM() {
        copy_warp_fragment_RF2SM<ldst, value_t>(data(), smem_srm_ptr, lane_id);
    }
};

} // namespace flash
// <<< tensor.cuh
// >>> utils.h
// <<< utils.h

namespace flash {

template <int n, int K, bool double_buffer>
constexpr void static_assert_valid_load_k_fragments() {
    static_assert(((n & (n - 1)) == 0) && n != 1,
                  "load k is power of 2 and DNE 1");

    constexpr int max_frags = (double_buffer ? K / 2 : K) / 8;
    static_assert(n <= max_frags, "load k is <= max fragments");
}

template <FlashForwardKernelConfig cfg>
constexpr bool valid_config() {
    static_assert_valid_load_k_fragments<cfg.Q_mma_load_K_fragments, cfg.d_head,
                                         cfg.mma_double_buffer_loads>();
    static_assert_valid_load_k_fragments<cfg.K_mma_load_K_fragments, cfg.d_head,
                                         cfg.mma_double_buffer_loads>();
    static_assert_valid_load_k_fragments<cfg.V_mma_load_K_fragments, cfg.B_c,
                                         cfg.mma_double_buffer_loads>();

    static_assert((cfg.Q_mma_load_K_fragments == cfg.K_mma_load_K_fragments) ||
                  cfg.Q_mma_load_K_fragments == 0);

    return true;
}

template <FlashForwardKernelConfig CFG>
struct ForwardKernelTileShapes {
    static_assert(valid_config<CFG>());

    // The number of d_head tiles loaded and operated on by this thread
    // block.
    static constexpr int d_head_fragments = CFG.d_head / COLS_PER_FRAGMENT;
    static constexpr int d_head_accum_regs =
        d_head_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT;

    // The number of Q/O rows/tiles each warp independently loads and computes
    // on, which corresponds to a (B_r / n_warps, d_head) chunk.
    static constexpr int QO_rows_per_warp = CFG.B_r / CFG.n_warps;
    static constexpr int QO_fragments_per_warp =
        QO_rows_per_warp / ROWS_PER_FRAGMENT;

    // For a K/V block, each warp will independently load a chunk of the (B_c,
    // d_head), but perform computations on the entire block loaded by the
    // thread-block.

    // The number of K/V tiles that each warp operates on, which corresponds to
    // a (B_c, d_head) chunks.
    static constexpr int KV_calc_fragments = CFG.B_c / ROWS_PER_FRAGMENT;
    static constexpr int KV_calc_accum_regs =
        KV_calc_fragments * N_REGS_PER_F32_ACCUM_FRAGMENT;

    // The number of K/V tiles that each warp loads into smem, which corresponds
    // to a (B_c/n_warps, d_head) chunk.
    static constexpr int KV_ldst_fragments_per_warp =
        KV_calc_fragments / CFG.n_warps;
    static constexpr int KV_ldst_rows_per_warp =
        KV_ldst_fragments_per_warp * ROWS_PER_FRAGMENT;

    // # tiles to load during matmuls between mma instructions.
    static constexpr int Q_mma_load_K_fragments =
        CFG.Q_mma_load_K_fragments == 0 ? d_head_fragments
                                        : CFG.Q_mma_load_K_fragments;
    static constexpr int Q_mma_load_stages =
        (CFG.Q_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;

    static constexpr int K_mma_load_K_fragments =
        CFG.K_mma_load_K_fragments == 0 ? d_head_fragments
                                        : CFG.K_mma_load_K_fragments;
    static constexpr int K_mma_load_stages =
        (CFG.K_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;

    static constexpr int V_mma_load_K_fragments =
        CFG.V_mma_load_K_fragments == 0 ? KV_calc_fragments
                                        : CFG.V_mma_load_K_fragments;
    static constexpr int V_mma_load_stages =
        (CFG.V_mma_load_K_fragments > 0 && CFG.mma_double_buffer_loads) ? 2 : 1;
};

template <FlashForwardKernelConfig CFG>
struct StaticForwardKernelConfig {
    using accum_t = float;
    using value_t = typename std::conditional_t<CFG.dtype == torch::kBFloat16,
                                                nv_bfloat16, half>;
    using N = ForwardKernelTileShapes<CFG>;

    // Static configuration fields accessed from the original CFG
    static constexpr bool async_copy = CFG.async_copy;
    static constexpr int B_r = CFG.B_r;
    static constexpr int B_c = CFG.B_c;
    static constexpr int d_head = CFG.d_head;
    static constexpr bool eager_load_blocks = CFG.eager_load_blocks;
    static constexpr bool optimized_softmax = CFG.optimized_softmax;

    static constexpr LDSTCommon Common{CFG.swizzled, CFG.async_copy};

    static constexpr TensorLDSTConfig make_ldst_config(
        TileLayout GSM, TileLayout RF, bool transposed, int block_size,
        int warp_ldst_rows, bool compute_over_entire_block,
        bool load_entire_block_into_rf = true, int mma_load_stages = 1) {

        return TensorLDSTConfig{GSM,
                                RF,
                                Common,
                                transposed,
                                block_size,
                                CFG.d_head,
                                warp_ldst_rows,
                                compute_over_entire_block,
                                load_entire_block_into_rf,
                                mma_load_stages};
    }

    static constexpr TensorLDSTConfig Q_LDST =
        make_ldst_config({N::QO_fragments_per_warp, N::d_head_fragments},
                         {N::QO_fragments_per_warp, N::Q_mma_load_K_fragments},
                         false /*transposed*/, CFG.B_r, N::QO_rows_per_warp,
                         false /*compute_over_entire_block*/,
                         CFG.Q_mma_load_K_fragments == 0, N::Q_mma_load_stages);
    using Q_t = MatrixLDST<Q_LDST, value_t>;

    static constexpr TensorLDSTConfig K_LDST = make_ldst_config(
        {N::KV_ldst_fragments_per_warp, N::d_head_fragments},
        {N::KV_calc_fragments, N::K_mma_load_K_fragments}, false /*transposed*/,
        CFG.B_c, N::KV_ldst_rows_per_warp, true /*compute_over_entire_block*/,
        CFG.K_mma_load_K_fragments == 0, N::K_mma_load_stages);
    using K_t = MatrixLDST<K_LDST, value_t>;

    static constexpr TensorLDSTConfig V_LDST = make_ldst_config(
        {N::KV_ldst_fragments_per_warp, N::d_head_fragments},
        {N::d_head_fragments, N::V_mma_load_K_fragments}, true /*transposed*/,
        CFG.B_c, N::KV_ldst_rows_per_warp, true /*compute_over_entire_block*/,
        CFG.V_mma_load_K_fragments == 0, N::V_mma_load_stages);
    using V_t = MatrixLDST<V_LDST, value_t>;

    static constexpr TensorLDSTConfig O_LDST =
        make_ldst_config({N::QO_fragments_per_warp, N::d_head_fragments},
                         {N::QO_fragments_per_warp, N::d_head_fragments},
                         false /*transposed*/, CFG.B_r, N::QO_rows_per_warp,
                         false /*compute_over_entire_block*/, true);
    using O_accum_t = MatrixLDST<O_LDST, accum_t>;
    using O_value_t = MatrixLDST<O_LDST, value_t>;

    // S/P is kept entirely in the RF during the entire duration of the kernel.
    static constexpr TensorLDSTConfig S_LDST = make_ldst_config(
        {N::QO_fragments_per_warp, N::KV_calc_fragments},
        {N::QO_fragments_per_warp, N::KV_calc_fragments}, CFG.B_r, false,
        0 /* only stored in RF, not smem or gmem */,
        false /*compute_over_entire_block*/);
    using S_accum_t = MatrixLDST<S_LDST, accum_t>;
    using P_value_t = MatrixLDST<S_LDST, value_t>;

    using S_QK_GEMM = GEMM<Q_t, K_t, S_accum_t, N::d_head_fragments,
                           constexpr_min(N::Q_mma_load_K_fragments,
                                         N::K_mma_load_K_fragments),
                           value_t>;
    using O_PV_GEMM = GEMM<P_value_t, V_t, O_accum_t, N::KV_calc_fragments,
                           N::V_mma_load_K_fragments, value_t>;

    using row_statistics_t = RFVector<accum_t, N::QO_fragments_per_warp>;
};

} // namespace flash
// <<< static_kernel_configuration.cuh

namespace flash {

template <typename Kernel>
__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {
    using accum_t = float;
    using index_t = int64_t;
    using N = typename Kernel::N;

    using value_t = typename Kernel::value_t;
    using Q_t = typename Kernel::Q_t;
    using K_t = typename Kernel::K_t;
    using V_t = typename Kernel::V_t;
    constexpr int async = Kernel::async_copy;

    // We initialize a CTA for each sample, seq tile, and head.
    const int sample = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq_block = blockIdx.x;

    const index_t gmem_seq_stride = args.seq_stride;

    const index_t sample_head_offset =
        sample * args.batch_stride + head * args.head_stride;
    // We only read/write one block for Q and O.
    // These offsets are the same for the whole thread-block.
    const index_t QO_gmem_block_offset =
        sample_head_offset + q_seq_block * Kernel::B_r * gmem_seq_stride;
    // We read the entire key sequence.
    const index_t KV_gmem_block_offset = sample_head_offset;

    value_t *gmem_Q = &static_cast<value_t *>(args.Q)[QO_gmem_block_offset];
    value_t *gmem_O = &static_cast<value_t *>(args.O)[QO_gmem_block_offset];
    value_t *gmem_K = &static_cast<value_t *>(args.K)[KV_gmem_block_offset];
    value_t *gmem_V = &static_cast<value_t *>(args.V)[KV_gmem_block_offset];

    extern __shared__ __align__(16) char ch_smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(ch_smem);
    value_t *smem_O = smem_Q;
    value_t *smem_K = &smem_Q[Kernel::B_r * Kernel::d_head];
    value_t *smem_V = &smem_K[Kernel::B_c * Kernel::d_head];

    // Pointers to the K&V locations in smem that the warp copies to.
    Q_t Q(gmem_Q, gmem_seq_stride, smem_Q);
    K_t K(gmem_K, gmem_seq_stride, smem_K);
    V_t V(gmem_V, gmem_seq_stride, smem_V);
    // S is only stored in registers.
    typename Kernel::S_accum_t S_accum(nullptr, -1, nullptr);
    // P is only stored in registers.
    typename Kernel::P_value_t P_b16(nullptr, -1, nullptr);
    // The accumulator for O is only kept in registers. At the end of the
    // kernel, it is then converted into a 16-bit type and then copied into
    // gmem.
    typename Kernel::O_accum_t O_accum(nullptr, -1, nullptr);
    typename Kernel::O_value_t O_b16(gmem_O, gmem_seq_stride, smem_O);

    // Start the async copy of the Q and K tiles.
    Q.copy_GM2SM();
    cp_async_commit<async>();
    if constexpr (Kernel::eager_load_blocks) {
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit<async>();
    }

    O_accum.zero();

    // Initialize softmax_scale, m, and l.
    const accum_t softmax_scale = rsqrt(static_cast<accum_t>(Kernel::d_head)) *
                                  (Kernel::optimized_softmax ? M_LOG2E : 1.0);
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();
    accum_t m[N::QO_fragments_per_warp];
    accum_t l[N::QO_fragments_per_warp];
    FA_UNROLL
    for (int q = 0; q < N::QO_fragments_per_warp; ++q) {
        m[q] = neg_inf;
        l[q] = 0.0;
    }

    if constexpr (Q_t::load_entire_block_into_rf) {
        if constexpr (Kernel::eager_load_blocks) {
            // We only wait for the Q block to finish loading.
            cp_async_wait<1, async>();
        } else {
            cp_async_wait<0, async>();
        }
        // We need the __syncwarp() in addition to the cp_async_wait()
        // because cp_async_wait() only blocks until the current thread has
        // finished loading. The entire warp will read this block from
        // smem, so we need to wait on a warp-wide barrier.
        // For K and V, we will need a __syncthread() instead.
        __syncwarp();
        Q.copy_SM2RF();
    }

    for (int j = 0; j < args.n_KV_blocks; ++j) {
        if constexpr (!Kernel::eager_load_blocks) {
            K.copy_GM2SM();
            K.advance_gmem_block();
            cp_async_commit<async>();
        }
        // Initialize the registers for S to 0.
        S_accum.zero();

        // Block until we've copied the K block-tile for this iteration into
        // shared memory.
        cp_async_wait<0, async>();
        // After this barrier, it is safe to load the next V block, because all
        // warps have done the previous PV matmul.
        __syncthreads();

        if constexpr (Kernel::eager_load_blocks) {
            // Start the (async) copy for the V matrix from gmem to smem but
            // do not wait until after the S=QK matmul.
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<async>();
        }
        if constexpr (K_t::load_entire_block_into_rf) {
            K.copy_SM2RF();
        }

        matmul<Kernel::S_QK_GEMM>(Q, K, S_accum);
        cp_async_wait<0, async>();
        // After this barrier, it is safe to load the next block of K.
        __syncthreads();

        if constexpr (Kernel::eager_load_blocks) {
            // Start the async copy for the next K block-tile from gmem to
            // smem, but do not wait for the copy until the next iteration
            // when we need it.
            if (j < args.n_KV_blocks - 1) {
                K.copy_GM2SM();
                K.advance_gmem_block();
                cp_async_commit<async>();
            }
        }

        // Online softmax
        accum_t m_next[N::QO_fragments_per_warp];
        if constexpr (!Kernel::optimized_softmax) {
            scale_S_accum(S_accum.data(), softmax_scale);
        }
        calc_row_max(S_accum.data(), m_next, m);
        scale_l_O<Kernel::optimized_softmax>(m_next, m, l, O_accum.data(),
                                             softmax_scale);
        exponentiate_tensor<Kernel::optimized_softmax>(S_accum.data(), m_next,
                                                       softmax_scale);
        update_row_exp_sum(S_accum.data(), l);

        // Convert the S accumulator block into P fp16 input block.
        convert_to_16_bit_dtype<value_t>(S_accum.data(), P_b16.data());

        if constexpr (!Kernel::eager_load_blocks) {
            // Load V from gmem to smem and block until it is done.
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<async>();
            cp_async_wait<0, async>();
            __syncthreads();
        }

        if constexpr (V_t::load_entire_block_into_rf) {
            V.copy_SM2RF();
        }

        matmul<typename Kernel::O_PV_GEMM>(P_b16, V, O_accum);
    }

    final_softmax_normalization(O_accum.data(), l);

    convert_to_16_bit_dtype<value_t>(O_accum.data(), O_b16.data());
    // Instead of writing directly to gmem, we write to smem as an intermediary
    // step. This allows us to
    // - use 16B vectorized stores, as opposed to 4B stores
    // - fully coalesce our stores
    //   - each warp can store 4x128B aligned lines (512B/warp) instead
    //   of 8x16B uncoalesced rows (128B/warp)
    O_b16.copy_RF2SM();

    // Wait until all threads in the same warp have written to smem.
    // We do not need __syncthreads() here because the warps operate on
    // independent chunks of O.
    __syncwarp();

    // Copy the final O tile from smem to gmem.
    O_b16.copy_SM2GM();
}

} // namespace flash
// <<< forward_kernel.cuh

namespace flash {

typedef void (*forward_kernel_fn)(const ForwardKernelArgs);

std::map<FlashForwardKernelConfig, forward_kernel_fn>
    forward_kernels = {
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (FP16, 128, 64, 64, 4): async+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, false, 0, 0, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, false, 0, 0, 0, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+swizzled+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, true, 0, 0, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, true, 0, 0, 0, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false}>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false}>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true}>>}
};
} // namespace flash// <<< flash_kernels.cuh

using namespace flash;

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

decltype(auto)
flash_attention_forward(const py::object &py_cfg, const torch::Tensor &TQ,
                        const torch::Tensor &TK, const torch::Tensor &TV,
                        std::optional<at::Tensor> &out_, bool benchmark) {
    CHECK_INPUT(TQ);
    CHECK_INPUT(TK);
    CHECK_INPUT(TV);

    at::cuda::CUDAGuard device_guard{TQ.device()};
    const int compute_capability =
        cuda_device_compute_capability(TQ.device().index());
    TORCH_CHECK(compute_capability >= 80,
                "Flash Attention requires SM_80 or higher (current: SM_",
                compute_capability / 10, ".", compute_capability % 10, ")");

    // Check data types
    const auto Q_dtype = TQ.dtype();
    TORCH_CHECK(Q_dtype == torch::kFloat16 || Q_dtype == torch::kBFloat16,
                "Only fp16 and bf16 are supported");
    TORCH_CHECK(TK.dtype() == Q_dtype,
                "Input tensors must have the same data type");
    TORCH_CHECK(TV.dtype() == Q_dtype,
                "Input tensors must have the same data type");

    const auto d_head = TQ.size(3);
    const FlashForwardKernelConfig cfg{py_to_cpp_kernel_config(py_cfg)};
    TORCH_CHECK(forward_kernels.contains(cfg),
                "Kernel configuration was not found in flash_kernels.cuh");
    const auto kernel = forward_kernels[cfg];

    TORCH_CHECK(cfg.dtype == Q_dtype,
                "Kernel configuration dtype does not match input dtype");

    const auto batch_size = TQ.size(0);
    const auto seq_len = TQ.size(1);
    const auto n_heads = TQ.size(2);

    // Only supported configuration currently.
    TORCH_CHECK(TQ.sizes() == TK.sizes(),
                "Query and key tensors have same shape");
    TORCH_CHECK(TQ.sizes() == TV.sizes(),
                "Query and value tensors have same shape");

    const int B_r = cfg.B_r;
    const int B_c = cfg.B_c;
    TORCH_CHECK(seq_len % B_r == 0,
                "Only multiples of B_r are supported for seq_len Q currently");
    TORCH_CHECK(seq_len % B_c == 0,
                "Only multiples of B_c are supported for seq_len K currently");

    const auto batch_stride = TQ.stride(0);
    const auto seq_stride = TQ.stride(1);
    const auto head_stride = TQ.stride(2);

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

    const int n_Q_blocks = CEIL_DIV(seq_len, B_r);
    const int n_KV_blocks = CEIL_DIV(seq_len, B_c);
    const int n_threads = cfg.n_warps * WARP_SIZE;

    ForwardKernelArgs args{TQ.data_ptr(), TK.data_ptr(), TV.data_ptr(),
                           TO.data_ptr(), batch_stride,  seq_stride,
                           head_stride,   seq_len,       n_heads,
                           n_Q_blocks,    n_KV_blocks};

    dim3 blockDim(n_threads);
    dim3 gridDim{static_cast<uint>(n_Q_blocks), static_cast<uint>(n_heads),
                 static_cast<uint>(batch_size)};

    float runtime;
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_forward, py::arg("kernel_cfg"),
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("o"),
          py::arg("benchmark") = false, "Flash Attention forward (CUDA)");

    // Set kernel max dynamic smem on module initialization.
    for (const auto &[cfg, kernel] : forward_kernels) {
        int smem_used = cfg.smem_bytes();
        if (smem_used > 48 * 1024) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_used);
        }
    }
}