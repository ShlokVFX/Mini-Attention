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

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#ifdef FA_DEBUG
#define FA_UNROLL
#else
#define FA_UNROLL _Pragma("unroll")
#endif

#define FA_DEVICE __forceinline__ __device__
#define FA_DEVICE_CONSTEXPR __forceinline__ __device__ constexpr

#define N_HEADS 16

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

#define THR_COLS_PER_ACCUM_FRAGMENT 2

#define LDMATRIX_MAT_SIZE 8
#define ROWS_PER_FRAGMENT LDMATRIX_MAT_SIZE
#define COLS_PER_FRAGMENT LDMATRIX_MAT_SIZE

#define N_BUFFER_STAGES 2

#define GSMEM_THR_PER_ROW 8
#define SWIZZLE_TILE_SIZE 64

namespace flash {

struct alignas(16) uint128_t {
    uint64_t low;
    uint64_t high;
};

template <typename value_t>
constexpr bool is_supported_mma_input_type() {
    return std::is_same_v<value_t, half> ||
           std::is_same_v<value_t, nv_bfloat16>;
}

template <typename value_t>
constexpr bool is_supported_mma_output_type() {
    return std::is_same_v<value_t, float>;
}

template <typename value_t>
constexpr auto value_storage_type() {
    if constexpr (is_supported_mma_input_type<value_t>()) {
        return uint32_t{};
    } else if constexpr (is_supported_mma_output_type<value_t>()) {
        return float{};
    }
}

template <typename value_t>
constexpr auto value2_storage_type() {
    if constexpr (std::is_same_v<value_t, half>) {
        return half2{};
    } else if constexpr (std::is_same_v<value_t, nv_bfloat16>) {
        return nv_bfloat162{};
    } else if constexpr (std::is_same_v<value_t, float>) {
        return float2{};
    }
}

} // namespace flash// <<< common.h
// >>> debug.cuh

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

// >>> common.h
// <<< common.h
// >>> flash_attention.cuh
// <<< flash_attention.cuh
// >>> gemm.cuh

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

// >>> common.h
// <<< common.h
// >>> debug.cuh
// <<< debug.cuh
// >>> layout.cuh

#include <stdio.h>
// >>> common.h
// <<< common.h
// >>> debug.cuh
// <<< debug.cuh
// >>> utils.h

namespace flash {

constexpr int constexpr_min(int a, int b) { return (a < b) ? a : b; }

constexpr int constexpr_max(int a, int b) { return (a > b) ? a : b; }

constexpr int constexpr_log2_floor(int n) { return std::__bit_width(n) - 1; }

constexpr int binary_to_pm1(int b) { return 2 * b - 1; }

} // namespace flash
// <<< utils.h

namespace flash {

// This could be rewritten to be far more elegant with template
// metaprogramming, but I wanted to avoid that for this project so that
// the code could be somewhat more "readable".

template <int Row, int Col, int Tile = 0, int SwizzleSpace = 0>
struct SMemStride {
    FA_DEVICE_CONSTEXPR static int row() { return Row; }
    FA_DEVICE_CONSTEXPR static int col() { return Col; }
    FA_DEVICE_CONSTEXPR static int tile() { return Tile; }
    FA_DEVICE_CONSTEXPR static int swizzle_space() { return SwizzleSpace; }

    FA_DEVICE_CONSTEXPR static int crd2idx(int row_, int col_, int tile_,
                                           int swizzle_space_) {
        return row_ * row() + col_ * col() + tile_ * tile() +
               swizzle_space_ * swizzle_space();
    }
};

template <int Rows, int Cols, int Tiles = 1, int SwizzleSpaces = 1>
struct GSMemShape {
    FA_DEVICE_CONSTEXPR static int rows() { return Rows; }
    FA_DEVICE_CONSTEXPR static int cols() { return Cols; }
    FA_DEVICE_CONSTEXPR static int tiles() { return Tiles; }
    FA_DEVICE_CONSTEXPR static int swizzle_spaces() { return SwizzleSpaces; }
};

template <int col, int tile, int swizzle_space, typename index_t = int64_t>
struct GMemStride {
    // Only component of stride unknown at build time is row stride
    const index_t row;

    FA_DEVICE_CONSTEXPR index_t crd2idx(int row_, int col_, int tile_,
                                        int swizzle_space_) const {
        return row_ * row + col_ * col + tile_ * tile +
               swizzle_space_ * swizzle_space;
    }
};

// This is a stride specific for swizzling.
// TODO: consolidate with SMemStride
struct SwizzleStride {
    int s1;
    int s2;
    int s3;

    // This determines the iteration of the copy we're in.
    constexpr int offset_s2rmem(int iter) const {
        int i1 = (iter >> 1) & 1;
        int i2 = iter & 1;
        return i1 * s1 + i2 * s2;
    }

    constexpr int offset_r2smem(int iter) const {
        int i1 = (iter >> 2) & 1;
        int i2 = (iter >> 1) & 1;
        int i3 = iter & 1;
        return i1 * s1 + i2 * s2 + i3 * s3;
    }
};

template <int Row, int Col, int Tile, int OpRow, int OpCol>
struct RmemStride {
    // Public accessor methods
    FA_DEVICE_CONSTEXPR static int row() { return Row; }
    FA_DEVICE_CONSTEXPR static int col() { return Col; }
    FA_DEVICE_CONSTEXPR static int tile() { return Tile; }
    FA_DEVICE_CONSTEXPR static int op_row() { return OpRow; }
    FA_DEVICE_CONSTEXPR static int op_col() { return OpCol; }
};

template <int Rows, int Cols, int Tiles, int OpRows, int OpCols,
          bool op_tiling_removed = false>
struct RmemShape {

    // Public accessor methods
    FA_DEVICE_CONSTEXPR static int rows() {
        return op_tiling_removed ? Rows * OpRows : Rows;
    }

    FA_DEVICE_CONSTEXPR static int cols() {
        return op_tiling_removed ? Cols * OpCols : Cols;
    }

    FA_DEVICE_CONSTEXPR static int tiles() { return Tiles; }

    FA_DEVICE_CONSTEXPR static int op_rows() {
        return op_tiling_removed ? 1 : OpRows;
    }

    FA_DEVICE_CONSTEXPR static int op_cols() {
        return op_tiling_removed ? 1 : OpCols;
    }

    FA_DEVICE_CONSTEXPR static int op_size() {
        return op_tiling_removed ? 1 : OpRows * OpCols;
    }

    FA_DEVICE_CONSTEXPR static int tile_size() {
        return rows() * cols() * op_size();
    }

    FA_DEVICE_CONSTEXPR static int size() { return Tiles * tile_size(); }

  private:
    template <typename, typename, bool>
    friend struct RmemLayout;

    static constexpr int _op_rows = OpRows;
    static constexpr int _op_cols = OpCols;
};

template <typename Stride_, typename Shape_, bool OpTilingRemoved = false>
struct RmemLayout {
    using Stride = Stride_;
    using Shape = Shape_;
    static constexpr bool op_tiling_removed = OpTilingRemoved;

    FA_DEVICE_CONSTEXPR static auto layout_as_2x2_op_tiled() {
        if constexpr (Shape::op_rows() == 2 && Shape::op_cols() == 2) {
            return RmemLayout<Stride, Shape>{};
        } else {
            using NewShape =
                RmemShape<Shape::rows() / 2, Shape::cols(), Shape::tiles(),
                          Shape::op_rows() * 2, Shape::op_cols()>;
            using NewStride =
                RmemStride<Stride::row() * 2, Stride::col(), Stride::tile(),
                           Stride::op_row(), Stride::op_col()>;
            return RmemLayout<NewStride, NewShape>{};
        }
    }

    FA_DEVICE_CONSTEXPR static auto layout_as_type2() {
        using NewStride =
            RmemStride<Stride::row() / 2, Stride::col(), Stride::tile() / 2,
                       Stride::op_row(), Stride::op_col()>;
        using NewShape =
            RmemShape<Shape::rows(), Shape::cols() / 2, Shape::tiles(),
                      Shape::op_rows(), Shape::op_cols()>;
        return RmemLayout<NewStride, NewShape>{};
    }

    // This is a hacky way to make indexing into the accumulator block easier.
    // Since
    FA_DEVICE_CONSTEXPR static auto layout_with_op_tiling_removed() {
        using NewShape = RmemShape<Shape::rows(), Shape::cols(), Shape::tiles(),
                                   Shape::op_rows(), Shape::op_cols(), true>;
        return RmemLayout<Stride, NewShape, true>{};
    }

    FA_DEVICE_CONSTEXPR static int crd2idx(int row, int col, int tile,
                                           int op_row = 0, int op_col = 0) {

        if constexpr (op_tiling_removed) {
            op_row = row % Shape::_op_rows;
            row = row / Shape::_op_rows;
            op_col = col % Shape::_op_cols;
            col = col / Shape::_op_cols;
        }

        auto offset = tile * Stride::tile() + row * Stride::row() +
                      col * Stride::col() + op_row * Stride::op_row() +
                      op_col * Stride::op_col();
        return offset;
    }

    __device__ static void print() {
        using MyLayout = RmemLayout<Stride, Shape>;
        using type2_layout_t = decltype(MyLayout::layout_as_type2());
        using op_tiling_removed_layout_t =
            decltype(MyLayout::layout_with_op_tiling_removed());
        using layout_2x2_op_tiled_t =
            decltype(MyLayout::layout_as_2x2_op_tiled());

        printf("Original Layout:\n");
        printf("  Shape: rows=%d, cols=%d, tiles=%d, op_rows=%d, op_cols=%d\n",
               Shape::rows(), Shape::cols(), Shape::tiles(), Shape::op_rows(),
               Shape::op_cols());
        printf("  Actual Shape: _rows=%d, _cols=%d, _tiles=%d, _op_rows=%d, "
               "_op_cols=%d\n",
               Shape::_rows, Shape::_cols, Shape::_tiles, Shape::_op_rows,
               Shape::_op_cols);
        printf("  Stride: row=%d, col=%d, tile=%d, op_row=%d, op_col=%d\n\n",
               Stride::row(), Stride::col(), Stride::tile(), Stride::op_row(),
               Stride::op_col());

        printf("Type2 Layout:\n");
        printf("  Shape: rows=%d, cols=%d, tiles=%d, op_rows=%d, op_cols=%d\n",
               type2_layout_t::Shape::rows(), type2_layout_t::Shape::cols(),
               type2_layout_t::Shape::tiles(), type2_layout_t::Shape::op_rows(),
               type2_layout_t::Shape::op_cols());
        printf("  Actual Shape: _rows=%d, _cols=%d, _tiles=%d, _op_rows=%d, "
               "_op_cols=%d\n",
               type2_layout_t::Shape::_rows, type2_layout_t::Shape::_cols,
               type2_layout_t::Shape::_tiles, type2_layout_t::Shape::_op_rows,
               type2_layout_t::Shape::_op_cols);
        printf("  Stride: row=%d, col=%d, tile=%d, op_row=%d, op_col=%d\n\n",
               type2_layout_t::Stride::row(), type2_layout_t::Stride::col(),
               type2_layout_t::Stride::tile(), type2_layout_t::Stride::op_row(),
               type2_layout_t::Stride::op_col());

        printf("Layout with Op Tiling Removed:\n");
        printf("  Shape: rows=%d, cols=%d, tiles=%d, op_rows=%d, op_cols=%d\n",
               op_tiling_removed_layout_t::Shape::rows(),
               op_tiling_removed_layout_t::Shape::cols(),
               op_tiling_removed_layout_t::Shape::tiles(),
               op_tiling_removed_layout_t::Shape::op_rows(),
               op_tiling_removed_layout_t::Shape::op_cols());
        printf("  Actual Shape: _rows=%d, _cols=%d, _tiles=%d, _op_rows=%d, "
               "_op_cols=%d\n",
               op_tiling_removed_layout_t::Shape::_rows,
               op_tiling_removed_layout_t::Shape::_cols,
               op_tiling_removed_layout_t::Shape::_tiles,
               op_tiling_removed_layout_t::Shape::_op_rows,
               op_tiling_removed_layout_t::Shape::_op_cols);
        printf("  Stride: row=%d, col=%d, tile=%d, op_row=%d, op_col=%d\n\n",
               op_tiling_removed_layout_t::Stride::row(),
               op_tiling_removed_layout_t::Stride::col(),
               op_tiling_removed_layout_t::Stride::tile(),
               op_tiling_removed_layout_t::Stride::op_row(),
               op_tiling_removed_layout_t::Stride::op_col());

        printf("Layout as 2x2 Op Tiled:\n");
        printf("  Shape: rows=%d, cols=%d, tiles=%d, op_rows=%d, op_cols=%d\n",
               layout_2x2_op_tiled_t::Shape::rows(),
               layout_2x2_op_tiled_t::Shape::cols(),
               layout_2x2_op_tiled_t::Shape::tiles(),
               layout_2x2_op_tiled_t::Shape::op_rows(),
               layout_2x2_op_tiled_t::Shape::op_cols());
        printf("  Actual Shape: _rows=%d, _cols=%d, _tiles=%d, _op_rows=%d, "
               "_op_cols=%d\n",
               layout_2x2_op_tiled_t::Shape::_rows,
               layout_2x2_op_tiled_t::Shape::_cols,
               layout_2x2_op_tiled_t::Shape::_tiles,
               layout_2x2_op_tiled_t::Shape::_op_rows,
               layout_2x2_op_tiled_t::Shape::_op_cols);
        printf("  Stride: row=%d, col=%d, tile=%d, op_row=%d, op_col=%d\n\n",
               layout_2x2_op_tiled_t::Stride::row(),
               layout_2x2_op_tiled_t::Stride::col(),
               layout_2x2_op_tiled_t::Stride::tile(),
               layout_2x2_op_tiled_t::Stride::op_row(),
               layout_2x2_op_tiled_t::Stride::op_col());
    }
};

template <typename Shape, bool op_row_major>
constexpr auto stride_for_shape() {
    // We assume the outer shape is row major.
    constexpr int tile_stride = Shape::tile_size();
    constexpr int row_stride = Shape::op_size() * Shape::cols();
    constexpr int col_stride = Shape::op_size();
    constexpr int op_row_stride =
        op_row_major ? (Shape::op_rows() == 1 ? row_stride : Shape::op_cols())
                     : 1;
    constexpr int op_col_stride = op_row_major ? 1 : Shape::op_rows();
    return RmemStride<row_stride, col_stride, tile_stride, op_row_stride,
                      op_col_stride>{};
}

} // namespace flash
// <<< layout.cuh
// >>> ptx_functions.cuh

#include <cuda_bf16.h>
#include <cuda_fp16.h>
// >>> common.h
// <<< common.h

namespace flash {

FA_DEVICE void cp_async_commit() { asm volatile("cp.async.commit_group;"); }

template <int ngroups>
FA_DEVICE void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(ngroups));
}

FA_DEVICE void cp_async_commit_and_wait_all() {
    cp_async_commit();
    cp_async_wait<0>();
}

template <int size, typename T>
FA_DEVICE void cp_async(T *smem_to, T *gmem_from) {
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_to);
    // The .cg option bypasses the L1 cache.
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
                 :
                 : "r"(smem_ptr), "l"(gmem_from), "n"(size));
}

template <bool transpose, typename T>
FA_DEVICE void ldmatrix_x4(T *load_from, uint32_t &a1, uint32_t &a2,
                           uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    if constexpr (transpose) {
        asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16"
                     "{%0, %1, %2, %3}, [%4];"
                     : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                     : "r"(smem_ptr));
    } else {
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                     "{%0, %1, %2, %3}, [%4];"
                     : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                     : "r"(smem_ptr));
    }
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

//
// st functions
//
// Adapted from cccl:
// https://github.com/NVIDIA/cccl/blob/29e6e2fc0a2fcaf3e8b2451fe89e3a40edf8c2b0/libcudacxx/include/cuda/__ptx/instructions/generated/st.h
//

template <typename _B128>
FA_DEVICE void st_global(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");

    asm volatile("{\n\t .reg .b128 B128_src; \n\t"
                 "mov.b128 B128_src, {%1, %2}; \n"
                 "st.global.b128 [%0], B128_src;\n\t"
                 "}"
                 :
                 : "l"(__addr), "l"((*reinterpret_cast<longlong2 *>(&__src)).x),
                   "l"((*reinterpret_cast<longlong2 *>(&__src)).y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_relaxed_gpu(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");

    asm volatile("{\n\t .reg .b128 B128_src; \n\t"
                 "mov.b128 B128_src, {%1, %2}; \n"
                 "st.relaxed.gpu.global.b128 [%0], B128_src;\n\t"
                 "}"
                 :
                 : "l"(__addr), "l"((*reinterpret_cast<longlong2 *>(&__src)).x),
                   "l"((*reinterpret_cast<longlong2 *>(&__src)).y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_relaxed_sys(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");

    asm volatile("{\n\t .reg .b128 B128_src; \n\t"
                 "mov.b128 B128_src, {%1, %2}; \n"
                 "st.relaxed.sys.global.b128 [%0], B128_src;\n\t"
                 "}"
                 :
                 : "l"(__addr), "l"((*reinterpret_cast<longlong2 *>(&__src)).x),
                   "l"((*reinterpret_cast<longlong2 *>(&__src)).y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_ef(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");

    asm volatile("{\n\t .reg .b128 B128_src; \n\t"
                 "mov.b128 B128_src, {%1, %2}; \n"
                 "st.global.L1::evict_first.b128 [%0], B128_src;\n\t"
                 "}"
                 :
                 : "l"(__addr), "l"((*reinterpret_cast<longlong2 *>(&__src)).x),
                   "l"((*reinterpret_cast<longlong2 *>(&__src)).y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_ef_relaxed_gpu(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");

    asm volatile("{\n\t .reg .b128 B128_src; \n\t"
                 "mov.b128 B128_src, {%1, %2}; \n"
                 "st.relaxed.gpu.global.L1::evict_first.b128 [%0], "
                 "B128_src;\n\t"
                 "}"
                 :
                 : "l"(__addr), "l"((*reinterpret_cast<longlong2 *>(&__src)).x),
                   "l"((*reinterpret_cast<longlong2 *>(&__src)).y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_ef_relaxed_sys(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");

    asm volatile("{\n\t .reg .b128 B128_src; \n\t"
                 "mov.b128 B128_src, {%1, %2}; \n"
                 "st.relaxed.sys.global.L1::evict_first.b128 [%0], "
                 "B128_src;\n\t"
                 "}"
                 :
                 : "l"(__addr), "l"((*reinterpret_cast<longlong2 *>(&__src)).x),
                   "l"((*reinterpret_cast<longlong2 *>(&__src)).y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_na(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");

    asm volatile("{\n\t .reg .b128 B128_src; \n\t"
                 "mov.b128 B128_src, {%1, %2}; \n"
                 "st.global.L1::no_allocate.b128 [%0], B128_src;\n\t"
                 "}"
                 :
                 : "l"(__addr), "l"((*reinterpret_cast<longlong2 *>(&__src)).x),
                   "l"((*reinterpret_cast<longlong2 *>(&__src)).y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_na_relaxed_gpu(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");

    asm volatile("{\n\t .reg .b128 B128_src; \n\t"
                 "mov.b128 B128_src, {%1, %2}; \n"
                 "st.relaxed.gpu.global.L1::no_allocate.b128 [%0], "
                 "B128_src;\n\t"
                 "}"
                 :
                 : "l"(__addr), "l"((*reinterpret_cast<longlong2 *>(&__src)).x),
                   "l"((*reinterpret_cast<longlong2 *>(&__src)).y)
                 : "memory");
}

template <typename _B128>
FA_DEVICE void st_global_na_relaxed_sys(_B128 *__addr, _B128 __src) {
    static_assert(sizeof(_B128) == 16, "");

    asm volatile("{\n\t .reg .b128 B128_src; \n\t"
                 "mov.b128 B128_src, {%1, %2}; \n"
                 "st.relaxed.sys.global.L1::no_allocate.b128 [%0], "
                 "B128_src;\n\t"
                 "}"
                 :
                 : "l"(__addr), "l"((*reinterpret_cast<longlong2 *>(&__src)).x),
                   "l"((*reinterpret_cast<longlong2 *>(&__src)).y)
                 : "memory");
}

} // namespace flash// <<< ptx_functions.cuh
// >>> utils.h
// <<< utils.h

namespace flash {

// Dimensions of the mma instruction we're using
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define MMA_M_FRAGMENTS_PER_ITER 2 // (MMA_M / LDMATRIX_MAT_SIZE)
#define MMA_N_FRAGMENTS_PER_ITER 1 // (MMA_N / LDMATRIX_MAT_SIZE)
#define MMA_K_FRAGMENTS_PER_ITER 2 // (MMA_K / LDMATRIX_MAT_SIZE)

template <typename _A_t, typename _B_t, typename _C_t, int tiles,
          typename value_t_>
struct GEMM {
    using A_t = _A_t;
    using B_t = _B_t;
    using C_t = _C_t;
    using value_t = value_t_;

    static constexpr int Tiles = tiles;

    static constexpr bool DoubleBufferA =
        !A_t::load_entire_block_into_rf && A_t::rmem_tile_buffer_size > 1;
    static constexpr bool DoubleBufferB =
        !B_t::load_entire_block_into_rf && B_t::rmem_tile_buffer_size > 1;
    static constexpr bool DoubleBuffer = DoubleBufferA || DoubleBufferB;
};

// warp_fragment_mma_f32_accum
// A has shape (M, 2, tiles)
// B has shape (N, 2, tiles)
// C has shape (M, N, 1)
template <typename value_t, typename A_t, typename B_t, typename C_t>
FA_DEVICE_CONSTEXPR void warp_fragment_mma_f32_accum(A_t &A, B_t &B, C_t &C,
                                                     const int &tile) {

    static_assert(is_supported_mma_input_type<typename A_t::value_t>(),
                  "A must be a half or bfloat16 tensor");
    static_assert(is_supported_mma_input_type<typename B_t::value_t>(),
                  "B must be a half or bfloat16 tensor");
    static_assert(std::is_same_v<typename C_t::value_t, float>,
                  "C must be a float tensor");
    static_assert(A_t::Shape::tiles() == B_t::Shape::tiles(),
                  "A and B must have the same number of tiles");
    static_assert(A_t::Shape::cols() == 1 && B_t::Shape::cols() == 1,
                  "A and B must have a tile size of 1 op tile");
    static_assert(A_t::Shape::rows() == C_t::Shape::rows(),
                  "A and C must have the same M shape");
    // We divide by 2 here because C_t contains of 2x2 op tiles, while B_t
    // contains 1x2 op tiles.
    static_assert(B_t::Shape::rows() / 2 ==
                      C_t::Shape::cols() / THR_COLS_PER_ACCUM_FRAGMENT,
                  "B and C must have the same N shape");
    auto A_uint = A.view();
    auto B_uint = B.view();
    auto C_view = C.view();
    constexpr int M = decltype(A_uint)::Shape::rows();
    constexpr int N = decltype(B_uint)::Shape::rows();

    FA_UNROLL
    for (int n = 0; n < N; ++n) {
        FA_UNROLL
        for (int m = 0; m < M; ++m) {
            int ms = (n & 1) ? M - m - 1 : m;
            mma_m16n8k16_f32_accum<value_t>(
                C_view(ms, n, 0, 0, 0), C_view(ms, n, 0, 0, 1),
                C_view(ms, n, 0, 1, 0), C_view(ms, n, 0, 1, 1),
                A_uint(ms, 0, tile, 0, 0), A_uint(ms, 0, tile, 1, 0),
                A_uint(ms, 0, tile, 0, 1), A_uint(ms, 0, tile, 1, 1),
                B_uint(n, 0, tile, 0, 0), B_uint(n, 0, tile, 0, 1),
                C_view(ms, n, 0, 0, 0), C_view(ms, n, 0, 0, 1),
                C_view(ms, n, 0, 1, 0), C_view(ms, n, 0, 1, 1));
        }
    }
}

template <typename GEMM>
FA_DEVICE_CONSTEXPR void matmul(typename GEMM::A_t &A, typename GEMM::B_t &B,
                                typename GEMM::C_t &C) {
    using A_t = typename GEMM::A_t;
    using B_t = typename GEMM::B_t;
    using value_t = typename GEMM::value_t;

    if constexpr (GEMM::DoubleBuffer) {
        if constexpr (GEMM::DoubleBufferA) {
            A.copy_SM2RF(0);
        }
        if constexpr (GEMM::DoubleBufferB) {
            B.copy_SM2RF(0);
        }
    }

    FA_UNROLL
    for (int tile = 0; tile < GEMM::Tiles; ++tile) {
        if constexpr (!A_t::load_entire_block_into_rf ||
                      !B_t::load_entire_block_into_rf) {
            int load_tile = tile + (GEMM::DoubleBuffer ? 1 : 0);
            if (load_tile < GEMM::Tiles) {
                if constexpr (!A_t::load_entire_block_into_rf) {
                    A.copy_SM2RF(load_tile);
                }
                if constexpr (!B_t::load_entire_block_into_rf) {
                    B.copy_SM2RF(load_tile);
                }
            }
        }

        // Perform tile-wise outer products.
        warp_fragment_mma_f32_accum<value_t>(A, B, C, tile);
    }
}

} // namespace flash
// <<< gemm.cuh
// >>> layout.cuh
// <<< layout.cuh
// >>> load_store.cuh

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <tuple>

// >>> common.h
// <<< common.h
// >>> debug.cuh
// <<< debug.cuh
// >>> layout.cuh
// <<< layout.cuh
// >>> ptx_functions.cuh
// <<< ptx_functions.cuh
// >>> swizzling.cuh

// >>> common.h
// <<< common.h
// >>> utils.h
// <<< utils.h

namespace flash {

// Adapted from https://leimao.github.io/blog/CuTe-Swizzle/.
template <int BBits = 3, int MBase = 0, int SShift = 3>
struct CuteSwizzle {
    static constexpr int mbase = MBase;
    static constexpr int mask_bits = BBits;
    static constexpr int mask_shift = SShift;

    static constexpr int bit_mask = (1 << mask_bits) - 1;
    static constexpr int yy_mask = bit_mask << (mbase + mask_shift);
    static constexpr int yy_mask_lowest_bit = yy_mask & -yy_mask;

    FA_DEVICE_CONSTEXPR static auto apply(int const &offset) {
        const int row_shifted = (offset & yy_mask) >> mask_shift;
        return offset ^ row_shifted;
    }
};

struct NoSwizzle {
    FA_DEVICE_CONSTEXPR static auto apply(int const &offset) { return offset; }
};

} // namespace flash// <<< swizzling.cuh
// >>> utils.h
// <<< utils.h

namespace flash {

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
        st_global_na_relaxed_sys(reinterpret_cast<uint4 *>(gmem),
                                 reinterpret_cast<uint4 *>(smem)[0]);
    }
};

template <typename ShapeT, int TileBufferSize, bool LoadEntireBlockIntoRF,
          bool RowMajorOpTile = true>
struct RmemLdstConfig {
    using rmem_shape = ShapeT;
    static constexpr int rmem_tile_buffer_size = TileBufferSize;
    static constexpr bool load_entire_block_into_rf = LoadEntireBlockIntoRF;
    static constexpr bool row_major_op_tile = RowMajorOpTile;
};

// TODO: cleanup
template <typename RmemConfig, bool Transposed, int WarpLdstRows,
          bool ComputeOverEntireBlock>
struct GSRMemLdstConfig {
    using rmem = RmemConfig;
    static constexpr bool transposed = Transposed;

    // This is the # of rows a warp in a thread-block independently
    // loads/stores. This is only used for Q and O.
    static constexpr int warp_ldst_rows = WarpLdstRows;

    // Whether not the warp will compute over the entire block.
    // This is false for (Q&O&S) and true for (K&V).
    static constexpr bool compute_over_entire_block = ComputeOverEntireBlock;
};

// TODO: create a constructor and use as an object
template <typename Swizzle_, typename OpStride_, typename TensorShape_,
          typename SmemStride_, typename GMemStride_,
          typename index_t = int64_t>
struct GSMemLdstConfig {
    using Swizzle = Swizzle_;
    using OpStride = OpStride_;
    using TensorShape = TensorShape_;
    using SmemStride = SmemStride_;
    using GMemStride = GMemStride_;

    using OpIters =
        GSMemShape<TensorShape::rows() / OpStride::row(),
                   TensorShape::cols() / OpStride::col(), 0,
                   TensorShape::swizzle_spaces() / OpStride::swizzle_space()>;

    static constexpr int thrs_per_row = 8;

    static constexpr int tid_to_thr_row(int tid) { return tid / thrs_per_row; }

    static constexpr int tid_to_thr_col(int tid) {
        return (tid % thrs_per_row) * COLS_PER_FRAGMENT;
    }

    static constexpr index_t gmem_thr_offset(int tid) {
        return GMemStride::crd2idx(tid_to_thr_row(tid), tid_to_thr_col(tid), 0,
                                   0);
    }

    static constexpr int smem_thr_offset(int tid) {
        return Swizzle::apply(tid_to_thr_row(tid) * SmemStride::row() +
                              tid_to_thr_col(tid) * SmemStride::col());
    }
};

// Copy a (B_r, d_head) or (B_c, d_head) block from gmem to smem or vice
// versa. Each warp independently loads a (seq_len_per_warp, d_head) block.
// Each inner iteration loads a (4, 64) tile, where each row is loaded by a
// group of 8 consecutive threads.
//
// Note that the swizzling is completely determined before this is called. In
// other words, the swizzling offset is fixed. This is because each iteration
// executes in a different "swizzle space", and our layout allows us to keep the
// same offsets for a thread between iterations.
template <typename op, /* either GM2SM_async or SM2GM */
          typename Cfg, typename value_t = half>
FA_DEVICE_CONSTEXPR void copy_block_GSM(value_t *gmem, value_t *smem) {
    FA_UNROLL
    for (int ss = 0; ss < Cfg::OpIters::swizzle_spaces(); ++ss) {
        FA_UNROLL
        for (int ir = 0; ir < Cfg::OpIters::rows(); ++ir) {
            FA_UNROLL
            for (int ic = 0; ic < Cfg::OpIters::cols(); ++ic) {
                int r = ir * Cfg::OpStride::row();
                int c = ic * Cfg::OpStride::col();
                int t = ss * Cfg::OpStride::swizzle_space();

                auto smem_idx = Cfg::SmemStride::crd2idx(r, c, 0, t);
                auto gmem_idx = Cfg::GMemStride::crd2idx(r, c, 0, t);
                op()(&gmem[gmem_idx], &smem[smem_idx]);
            }
        }
    }
}

template <typename Swizzle_, typename OpSmemStride_, typename OpRmemStride_,
          typename SmemStride_, typename SmemShape_,
          bool SmemRowMajorLdmatrix_ = false>
struct SRMemLdstConfig {
    using Swizzle = Swizzle_;
    using OpSmemStride = OpSmemStride_;
    using OpRmemStride = OpRmemStride_;
    using SmemStride = SmemStride_;
    using SmemShape = SmemShape_;
    static constexpr bool smem_row_major_ldmatrix = SmemRowMajorLdmatrix_;
    using OpIters =
        GSMemShape<SmemShape::rows() / OpSmemStride::row(),
                   SmemShape::cols() / OpSmemStride::col(),
                   SmemShape::tiles() / OpSmemStride::tile(),
                   SmemShape::swizzle_spaces() / OpSmemStride::swizzle_space()>;

    static constexpr int lane_to_thr_offset_s2rmem(int lane_id) {
        int thread_row, thread_col;
        if constexpr (!smem_row_major_ldmatrix) {
            thread_row = lane_id % 16;
            thread_col = (lane_id / 16) * COLS_PER_FRAGMENT;
        } else {
            thread_row = (lane_id % 8) + 8 * (lane_id / 16);
            thread_col = lane_id & 8; // first or second column
        }
        return Swizzle::apply(thread_row * SmemStride::row() +
                              thread_col * SmemStride::col());
    }

    static constexpr int lane_to_thr_offset_r2smem(int lane_id) {
        constexpr int thr_per_row = 4;
        constexpr int elems_per_thr = 2;
        int thread_row = lane_id / thr_per_row;
        int thread_col = (lane_id % thr_per_row) * elems_per_thr;
        return Swizzle::apply(thread_row * SmemStride::row() +
                              thread_col * SmemStride::col());
    }

    static constexpr SwizzleStride
    lane_to_thr_swizzle_stride_s2rmem(int lane_id) {
        if constexpr (std::is_same_v<Swizzle, NoSwizzle>) {
            return SwizzleStride{32, 16};
        } else {
            int base_swizzle_offset = lane_to_thr_offset_s2rmem(lane_id);
            // Determine the swizzle offsets
            int base_offset_cmp = Swizzle::yy_mask_lowest_bit << 1;
            int s1 = 32 * binary_to_pm1((base_swizzle_offset &
                                         (base_offset_cmp << 1)) == 0);
            int s2 = 16 * binary_to_pm1(
                              (base_swizzle_offset & base_offset_cmp) == 0);

            return SwizzleStride{s1, s2};
        }
    }

    static constexpr SwizzleStride
    lane_to_thr_swizzle_stride_r2smem(int lane_id) {
        if constexpr (std::is_same_v<Swizzle, NoSwizzle>) {
            return SwizzleStride{32, 16, 8};
        } else {
            int base_swizzle_offset = lane_to_thr_offset_r2smem(lane_id);
            // Determine the swizzle offsets
            int base_offset_cmp = Swizzle::yy_mask_lowest_bit;
            int s1 = 32 * binary_to_pm1((base_swizzle_offset &
                                         (base_offset_cmp << 2)) == 0);
            int s2 = 16 * binary_to_pm1((base_swizzle_offset &
                                         (base_offset_cmp << 1)) == 0);
            int s3 =
                8 * binary_to_pm1((base_swizzle_offset & base_offset_cmp) == 0);

            return SwizzleStride{s1, s2, s3};
        }
    }
};

// Loads matrix fragments in smem into registers.
// Each ldmatrix.x4 instruction loads a (16, 16) chunk, i.e. (2, 2) fragments.
// For this non-transposed version, the shape of the smem matches rmem, i.e.
// shape(rmem) = (r_r, r_c) = (s_r / 8, s_c / 8).
// This will be used to copy Q and K.
template <typename Cfg, typename RmemType, typename value_t>
FA_DEVICE_CONSTEXPR void
copy_warp_fragment_SM2RF(RmemType &rmem, value_t *smem,
                         const SwizzleStride &swizzle_stride, const int &tile) {
    static_assert(is_supported_mma_input_type<value_t>(),
                  "value_t must be half or bfloat16");
    static_assert(Cfg::OpIters::rows() == RmemType::ViewType2x2::Shape::rows() /
                                              Cfg::OpRmemStride::row(),
                  "OpIters.rows must be equal to RmemType::Shape.rows / "
                  "Cfg::OpRmemStride.row");
    static_assert(RmemType::Shape::cols() == 1,
                  "RmemType::Shape.cols must be 2");
    auto rmem_uint = rmem.view2x2();
    const int inner_tile = tile % Cfg::SmemShape::tiles();
    const int swizzle_space = tile / Cfg::SmemShape::tiles();
    const int swizzle_offset = swizzle_stride.offset_s2rmem(inner_tile);

    FA_UNROLL
    for (int ir = 0; ir < Cfg::OpIters::rows(); ++ir) {
        int sr = ir * Cfg::OpSmemStride::row();
        // tile and col are 0 because ss's taken care of by swizzle.
        int smem_offset =
            Cfg::SmemStride::crd2idx(sr, 0, 0, swizzle_space) + swizzle_offset;
        int rr = ir * Cfg::OpRmemStride::row();

        if constexpr (!Cfg::smem_row_major_ldmatrix) {
            ldmatrix_x4<false>(&smem[smem_offset], rmem_uint(rr, 0, tile, 0, 0),
                               rmem_uint(rr, 0, tile, 1, 0),
                               rmem_uint(rr, 0, tile, 0, 1),
                               rmem_uint(rr, 0, tile, 1, 1));
        } else {
            ldmatrix_x4<false>(&smem[smem_offset], rmem_uint(rr, 0, tile, 0, 0),
                               rmem_uint(rr, 0, tile, 0, 1),
                               rmem_uint(rr, 0, tile, 1, 0),
                               rmem_uint(rr, 0, tile, 1, 1));
        }
    }
}

// Loads matrix fragments in smem into registers.
// Each ldmatrix.x4 instruction loads a (16, 16) chunk, i.e. (2, 2) fragments.
// For this transposed version, the shape of the smem matches the transpose of
// rmem, i.e. shape(rmem) = (r_r, r_c) = (s_c / 8, s_r / 8).
// This will be used to copy V.
template <typename Cfg, typename RmemType, typename value_t>
FA_DEVICE_CONSTEXPR void
copy_warp_fragment_transposed_SM2RF(RmemType &rmem, value_t *smem,
                                    const SwizzleStride &swizzle_stride,
                                    const int &tile) {
    static_assert(is_supported_mma_input_type<value_t>(),
                  "value_t must be half or bfloat16");
    static_assert(RmemType::Shape::cols() == 1,
                  "RmemType::Shape.cols must be 2");
    static_assert(Cfg::OpIters::tiles() * Cfg::OpIters::swizzle_spaces() ==
                      RmemType::ViewType2x2::Shape::rows() /
                          Cfg::OpRmemStride::col(),
                  "OpIters.tiles * OpIters.swizzle_spaces must be equal to "
                  "RmemType::Shape.rows / Cfg::OpRmemStride.col");
    auto rmem_uint = rmem.view2x2();

    FA_UNROLL
    for (int ss = 0; ss < Cfg::OpIters::swizzle_spaces(); ++ss) {
        FA_UNROLL
        for (int ic = 0; ic < Cfg::OpIters::tiles(); ++ic) {
            auto smem_idx = Cfg::SmemStride::crd2idx(0, 0, tile, ss) +
                            swizzle_stride.offset_s2rmem(ic);
            int rr =
                ss * Cfg::SmemShape::tiles() + ic * Cfg::OpRmemStride::row();
            ldmatrix_x4<true>(&smem[smem_idx], rmem_uint(rr, 0, tile, 0, 0),
                              rmem_uint(rr, 0, tile, 0, 1),
                              rmem_uint(rr, 0, tile, 1, 0),
                              rmem_uint(rr, 0, tile, 1, 1));
        }
    }
}

// Copies matrix fragments in rmem to smem.
// Each iteration of the inner loop copies a (8, 8) tile, i.e. a single
// fragment. This will be used to copy O.
template <typename Cfg, typename RmemType, typename value_t>
FA_DEVICE_CONSTEXPR void
copy_warp_fragment_RF2SM(RmemType &rmem, value_t *smem,
                         const SwizzleStride &swizzle_stride) {
    static_assert(is_supported_mma_input_type<value_t>(),
                  "value_t must be half or bfloat16");
    auto rmem_uint = rmem.view();

    FA_UNROLL
    for (int ss = 0; ss < Cfg::OpIters::swizzle_spaces(); ++ss) {
        FA_UNROLL
        for (int it = 0; it < Cfg::OpIters::tiles(); ++it) {
            FA_UNROLL
            for (int ir = 0; ir < Cfg::OpIters::rows(); ++ir) {
                int sr = ir * Cfg::OpSmemStride::row();
                auto smem_idx = Cfg::SmemStride::crd2idx(sr, 0, 0, ss) +
                                swizzle_stride.offset_r2smem(it);

                reinterpret_cast<uint32_t *>(&smem[smem_idx])[0] =
                    rmem_uint(ir * Cfg::OpRmemStride::row(),
                              it * Cfg::OpRmemStride::col() +
                                  ss * Cfg::SmemShape::tiles());
            }
        }
    }
}

template <typename SrcType, typename DstType>
FA_DEVICE_CONSTEXPR void convert_to_16_bit_dtype(SrcType &src_view,
                                                 DstType &dst_view) {
    static_assert(std::is_same_v<typename SrcType::value_t, float>,
                  "Input tensor must be float type");
    static_assert(std::is_same_v<typename DstType::value_t, half> ||
                      std::is_same_v<typename DstType::value_t, nv_bfloat16>,
                  "Output tensor must be half or bfloat16 type");
    using value_t = typename DstType::value_t;

    auto src = src_view.with_op_tiling_removed();
    auto dst2 = dst_view.with_op_tiling_removed().as_type2();
    using SrcShape = decltype(src)::Layout::Shape;
    using DstShape = decltype(dst2)::Layout::Shape;

    static_assert(SrcShape::tiles() == 1, "Src must have 1 tile");
    static_assert(SrcShape::cols() * SrcShape::tiles() ==
                      DstShape::cols() * DstShape::tiles() * 2,
                  "A and B must have the same shape");
    static_assert(SrcShape::rows() == DstShape::rows(),
                  "A and B must have the same shape");

    FA_UNROLL
    for (int tile = 0; tile < DstShape::tiles(); ++tile) {
        int tile_offset = 2 * tile * DstShape::cols();
        FA_UNROLL
        for (int m = 0; m < DstShape::rows(); ++m) {
            FA_UNROLL
            for (int k = 0; k < DstShape::cols(); ++k) {
                int src_k = tile_offset + 2 * k;
                float2 src_val{src(m, src_k, 0), src(m, src_k + 1, 0)};
                if constexpr (std::is_same_v<value_t, half>) {
                    dst2(m, k, tile) = __float22half2_rn(src_val);
                } else {
                    dst2(m, k, tile) = __float22bfloat162_rn(src_val);
                }
            }
        }
    }
}

} // namespace flash
// <<< load_store.cuh
// >>> static_kernel_configuration.cuh

// >>> common.h
// <<< common.h
// >>> flash_attention.cuh
// <<< flash_attention.cuh
// >>> gemm.cuh
// <<< gemm.cuh
// >>> layout.cuh
// <<< layout.cuh
// >>> load_store.cuh
// <<< load_store.cuh
// >>> tensor.cuh

// >>> array.cuh

// >>> common.h
// <<< common.h

template <int N, typename value_t_>
struct Array {
    using value_t = value_t_;

    value_t _data[N];

    FA_DEVICE_CONSTEXPR value_t *data() { return _data; }
    FA_DEVICE_CONSTEXPR const value_t *data() const { return _data; }

    FA_DEVICE_CONSTEXPR void fill(value_t val) {
        FA_UNROLL
        for (int i = 0; i < N; ++i) {
            _data[i] = value_t(val);
        }
    }

    FA_DEVICE_CONSTEXPR void zero() { fill(0); }

    FA_DEVICE_CONSTEXPR value_t &operator[](size_t idx) { return _data[idx]; }
    FA_DEVICE_CONSTEXPR value_t operator[](size_t idx) const {
        return _data[idx];
    }

    FA_DEVICE_CONSTEXPR static size_t size() { return N; }

    template <typename Other>
    FA_DEVICE_CONSTEXPR void copy(const Other &other) {
        static_assert(std::is_same<value_t, typename Other::value_t>::value,
                      "Arrays must have the same value type");
        static_assert(N == Other::size(), "Arrays must have the same size");

        FA_UNROLL
        for (int i = 0; i < N; ++i) {
            _data[i] = other[i];
        }
    }
};

template <int N, typename value_t, int Alignment = 16>
struct __align__(Alignment) ArrayAligned : public Array<N, value_t> {};
// <<< array.cuh
// >>> common.h
// <<< common.h
// >>> debug.cuh
// <<< debug.cuh
// >>> load_store.cuh
// <<< load_store.cuh
// >>> tensor_view.cuh

#include <type_traits>
// >>> common.h
// <<< common.h
// >>> layout.cuh
// <<< layout.cuh

namespace flash {

// TensorView
//
// `value_t` contains the actual type
// `storage_t` contains the storage type.
// - for fp16 and bf16, this is `uint32_t`.
// - for float, this is `float`.
//
// The indexing is given by the shape of the storage_t, so fp16 and bf16 and
// compressed by 2. This could've been better designed to handle this.
// TODO: better way to handle input and accum dtypes
template <typename value_t_, typename Layout_,
          typename storage_t_ = decltype(value_storage_type<value_t_>())>
struct TensorView {
    static_assert(is_supported_mma_input_type<value_t_>() ||
                      is_supported_mma_output_type<value_t_>(),
                  "value_t must be half, nv_bfloat16, or float");
    using value_t = value_t_;
    using storage_t = storage_t_;
    using Layout = Layout_;
    using Shape = typename Layout::Shape;
    using Stride = typename Layout::Stride;

    storage_t *data;

    FA_DEVICE_CONSTEXPR TensorView(storage_t *data) : data(data) {}

    template <typename NewLayout>
    FA_DEVICE_CONSTEXPR TensorView<value_t, NewLayout> with_layout() {
        static_assert(NewLayout::Shape::size() == Shape::size(),
                      "Shape size mismatch");
        return TensorView<value_t, NewLayout>(data);
    }

    FA_DEVICE_CONSTEXPR auto as_type2() {
        if constexpr (is_supported_mma_input_type<value_t>()) {
            using storage_t2 = std::conditional_t<std::is_same_v<value_t, half>,
                                                  half2, nv_bfloat162>;
            return as_storage_type<storage_t2, Layout>();
        } else if constexpr (std::is_same_v<value_t, float>) {
            using NewLayout = decltype(Layout::layout_as_type2());

            return as_storage_type<float2, NewLayout>();
        }
    }

    FA_DEVICE_CONSTEXPR auto with_op_tiling_removed() {
        if constexpr (Layout::op_tiling_removed) {
            return *this;
        } else {
            using NewLayout = decltype(Layout::layout_with_op_tiling_removed());
            return with_layout<NewLayout>();
        }
    }

    FA_DEVICE_CONSTEXPR storage_t &operator()(size_t row, size_t col,
                                              size_t tile = 0,
                                              size_t op_row = 0,
                                              size_t op_col = 0) {
        return data[Layout::crd2idx(row, col, tile, op_row, op_col)];
    }
    FA_DEVICE_CONSTEXPR storage_t operator()(size_t row, size_t col,
                                             size_t tile = 0, size_t op_row = 0,
                                             size_t op_col = 0) const {
        return data[Layout::crd2idx(row, col, tile, op_row, op_col)];
    }

  private:
    template <typename T, typename NewLayout>
    FA_DEVICE_CONSTEXPR auto as_storage_type() {
        return TensorView<value_t, NewLayout, T>(reinterpret_cast<T *>(data));
    }
};

} // namespace flash// <<< tensor_view.cuh

namespace flash {

// TODO(sonny): refactor these structs
// - use tensorview for both gmem and smem and use swizzle
template <typename RmemConfig, typename value_t_, typename index_t = int64_t>
struct RmemBlockTensor {
    using Shape = typename RmemConfig::rmem_shape;
    using Stride =
        decltype(stride_for_shape<Shape, RmemConfig::row_major_op_tile>());
    using Layout = RmemLayout<Stride, Shape>;
    using Layout2x2 = decltype(Layout::layout_as_2x2_op_tiled());

    using value_t = value_t_;
    using storage_t = decltype(value_storage_type<value_t>());

    using ViewType = TensorView<value_t, Layout>;
    using ViewType2x2 = TensorView<value_t, Layout2x2>;

    static constexpr int StorageSize = Shape::size();

    static constexpr int rmem_tile_buffer_size =
        RmemConfig::rmem_tile_buffer_size;
    static constexpr bool load_entire_block_into_rf =
        RmemConfig::load_entire_block_into_rf;

    ArrayAligned<StorageSize, storage_t> _storage;

    FA_DEVICE_CONSTEXPR void zero() { _storage.zero(); }

    FA_DEVICE_CONSTEXPR ViewType view() { return ViewType(_storage.data()); }
    FA_DEVICE_CONSTEXPR ViewType2x2 view2x2() {
        return ViewType2x2(_storage.data());
    }
    FA_DEVICE_CONSTEXPR auto view_as_type2() { return view().as_type2(); }

    FA_DEVICE_CONSTEXPR auto view_with_op_tiling_removed() {
        return view().with_op_tiling_removed();
    }
};

template <typename GSRConfig, typename value_t_, typename gsmem, typename srmem,
          typename index_t = int64_t>
struct GSRBlockTensor
    : public RmemBlockTensor<typename GSRConfig::rmem, value_t_, index_t> {
    using value_t = value_t_;
    using GSMemShape = gsmem::TensorShape;
    using SmemStride = gsmem::SmemStride;
    using GMemStride = gsmem::GMemStride;
    using Base = RmemBlockTensor<typename GSRConfig::rmem, value_t, index_t>;
    using GM2SM_op = GM2SM_async<value_t>;

    using SM2GM_op = SM2GM<value_t>;

    // The location in memory that the warp reads from for Q, K, V from gmem to
    // smem and O for smem to gmem.

    // TODO: move these to tensorview
    value_t *gmem_ptr;

    // The location in memory that the warp writes to for Q, K, V from gmem
    // to smem and O for smem to gmem. It is offset to the specific position
    // that the thread reads.
    value_t *smem_gsmem_ptr;
    // The location in memory used to load fragments from smem to rmem. This is
    // different that the ptr for smem when copying from gmem to smem because
    // the threads load different values in a different pattern.
    value_t *smem_s2rmem_ptr;
    value_t *smem_r2smem_ptr;

    SwizzleStride s2rmem_swizzle_stride;
    SwizzleStride r2smem_swizzle_stride;

    FA_DEVICE GSRBlockTensor(value_t *gmem_block_ptr, value_t *_smem_ptr)
        : Base() {
        const int tid = threadIdx.x;

        // We increment the pointers to the exact location for the thread.
        gmem_ptr = gmem_block_ptr + gsmem::gmem_thr_offset(tid);
        smem_gsmem_ptr = _smem_ptr + gsmem::smem_thr_offset(tid);

        const int lane_id = tid % WARP_SIZE;
        const int warp_rank = tid / WARP_SIZE;
        s2rmem_swizzle_stride =
            srmem::lane_to_thr_swizzle_stride_s2rmem(lane_id);
        r2smem_swizzle_stride =
            srmem::lane_to_thr_swizzle_stride_r2smem(lane_id);

        auto warp_offset =
            GSRConfig::compute_over_entire_block
                ? 0
                : GSRConfig::warp_ldst_rows * warp_rank * SmemStride::row();
        auto smem_srmem_ptr = _smem_ptr + warp_offset;

        smem_s2rmem_ptr =
            smem_srmem_ptr + srmem::lane_to_thr_offset_s2rmem(lane_id);
        smem_r2smem_ptr =
            smem_srmem_ptr + srmem::lane_to_thr_offset_r2smem(lane_id);
    }

    FA_DEVICE_CONSTEXPR void copy_GM2SM(const int &block) {
        auto gmem_block_offset = block * GSMemShape::rows() * GMemStride::row();
        copy_block_GSM<GM2SM_op, gsmem>(gmem_ptr + gmem_block_offset,
                                        smem_gsmem_ptr);
    }

    FA_DEVICE_CONSTEXPR void copy_SM2GM() {
        copy_block_GSM<SM2GM_op, gsmem>(gmem_ptr, smem_gsmem_ptr);
    }

    FA_DEVICE_CONSTEXPR void copy_SM2RF(int tile = 0) {
        if constexpr (!GSRConfig::transposed) {
            copy_warp_fragment_SM2RF<srmem>(*this, smem_s2rmem_ptr,
                                            s2rmem_swizzle_stride, tile);
        } else {
            copy_warp_fragment_transposed_SM2RF<srmem>(
                *this, smem_s2rmem_ptr, s2rmem_swizzle_stride, tile);
        }
    }

    FA_DEVICE_CONSTEXPR void copy_SM2RF_all_tiles() {
        for (int tile = 0; tile < Base::Shape::tiles(); ++tile) {
            copy_SM2RF(tile);
        }
    }

    FA_DEVICE_CONSTEXPR void copy_RF2SM() {
        copy_warp_fragment_RF2SM<srmem>(*this, smem_r2smem_ptr,
                                        r2smem_swizzle_stride);
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

template <const FlashForwardKernelConfig &cfg>
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

    static constexpr int n_threads = CFG.n_warps * WARP_SIZE;

    // The number of d_head tiles loaded and operated on by this thread
    // block.
    static constexpr int d_head_fragments = CFG.d_head / COLS_PER_FRAGMENT;

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

    // The number of K/V tiles that each warp loads into smem, which corresponds
    // to a (B_c/n_warps, d_head) chunk.
    static constexpr int KV_ldst_fragments_per_warp =
        KV_calc_fragments / CFG.n_warps;
    static constexpr int KV_ldst_rows_per_warp =
        KV_ldst_fragments_per_warp * ROWS_PER_FRAGMENT;

    static constexpr int get_tile_fragments(int val, int default_val) {
        return val == 0 ? default_val : val;
    }

    static constexpr int QK_rmem_tile_fragments = get_tile_fragments(
        constexpr_max(CFG.Q_mma_load_K_fragments, CFG.K_mma_load_K_fragments),
        d_head_fragments);
    static constexpr int QK_rmem_tile_size =
        QK_rmem_tile_fragments * COLS_PER_FRAGMENT;
    static constexpr int QK_rmem_tiles =
        d_head_fragments / QK_rmem_tile_fragments;
    static constexpr int PV_rmem_tile_fragments =
        get_tile_fragments(CFG.V_mma_load_K_fragments, KV_calc_fragments);
    static constexpr int PV_rmem_tile_size =
        PV_rmem_tile_fragments * COLS_PER_FRAGMENT;
    static constexpr int PV_rmem_tiles =
        KV_calc_fragments / PV_rmem_tile_fragments;

    static constexpr int get_rmem_tile_buffer_size(int load_K_fragments,
                                                   int tiles) {
        if (load_K_fragments == 0) {
            return tiles;
        }
        return CFG.mma_double_buffer_loads ? 2 : 1;
    }

    static constexpr int Q_rmem_tile_buffer_size =
        get_rmem_tile_buffer_size(CFG.Q_mma_load_K_fragments, QK_rmem_tiles);

    static constexpr int K_rmem_tile_buffer_size =
        get_rmem_tile_buffer_size(CFG.K_mma_load_K_fragments, QK_rmem_tiles);

    static constexpr int V_rmem_tile_buffer_size =
        get_rmem_tile_buffer_size(CFG.V_mma_load_K_fragments, PV_rmem_tiles);
};

template <FlashForwardKernelConfig _CFG>
struct StaticForwardKernelConfig {
    static constexpr FlashForwardKernelConfig CFG = _CFG;

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

    static constexpr int SwizzleTileSize = SWIZZLE_TILE_SIZE;
    static constexpr int DHeadSwizzleTiles = d_head / SwizzleTileSize;
    static constexpr int B_c_SwizzleTiles = B_c / SwizzleTileSize;

    using OpGSMemStride =
        SMemStride<N::n_threads / GSMEM_THR_PER_ROW, SwizzleTileSize, 0, 1>;

    using OpS2RSmemStride = SMemStride<16, 16, 1, 1>;
    using OpS2RRmemStride = RmemStride<1, 1, 1, 0, 0>;

    using OpR2SSMemStride = SMemStride<8, 8, 1, 1>;
    using OpR2SRmemStride = RmemStride<1, 1, 1, 0, 0>;

    // TODO: make sure this is consistent with tile size.
    using SmemSwizzle_ =
        CuteSwizzle<3, 3,
                    constexpr_log2_floor(SWIZZLE_TILE_SIZE) -
                        constexpr_log2_floor(ELEMS_PER_VEC4_ACCESS)>;
    using SmemSwizzle =
        std::conditional_t<CFG.swizzled, SmemSwizzle_, NoSwizzle>;

    using GSMemShapeQO = GSMemShape<B_r, SwizzleTileSize, 1, DHeadSwizzleTiles>;
    using GSMemShapeKV = GSMemShape<B_c, SwizzleTileSize, 1, DHeadSwizzleTiles>;

    // does not include row stride
    using GMemStride = SMemStride<CFG.d_head * N_HEADS, 1, 0, SwizzleTileSize>;

    using GSSMemQOStride =
        SMemStride<SwizzleTileSize, 1, 0, B_r * SwizzleTileSize>;
    using GSSMemKVStride =
        SMemStride<SwizzleTileSize, 1, 0, B_c * SwizzleTileSize>;

    using GSMemLdstConfigQO =
        GSMemLdstConfig<SmemSwizzle, OpGSMemStride, GSMemShapeQO,
                        GSSMemQOStride, GMemStride>;
    using GSMemLdstConfigKV =
        GSMemLdstConfig<SmemSwizzle, OpGSMemStride, GSMemShapeKV,
                        GSSMemKVStride, GMemStride>;

    static constexpr int SRMemTileSize = 16;
    static constexpr int SRMemTileFragments = SRMemTileSize / COLS_PER_FRAGMENT;
    static constexpr int SRMemTilesDHead = d_head / SRMemTileSize;
    static constexpr int SRMemTilesDHeadPerSwizzleTile =
        SRMemTilesDHead / DHeadSwizzleTiles;
    static constexpr int SRMemFragmentsDHead =
        SRMemTilesDHead * SRMemTileFragments;

    static constexpr int SRMemTilesB_c = B_c / SRMemTileSize;
    static constexpr int SRMemTilesB_cPerSwizzleTile =
        SRMemTilesB_c / B_c_SwizzleTiles;
    static constexpr int SRMemFragmentsB_c = SRMemTilesB_c * SRMemTileFragments;

    static constexpr int RSMemTileSize = 8;
    static constexpr int RSMemTilesDHead = d_head / RSMemTileSize;
    static constexpr int RSMemTilesDHeadPerSwizzleTile =
        RSMemTilesDHead / DHeadSwizzleTiles;

    using S2RSmemShapeQ =
        GSMemShape<N::QO_rows_per_warp, SRMemTileSize,
                   SRMemTilesDHeadPerSwizzleTile, DHeadSwizzleTiles>;
    using S2RSmemStrideQ = SMemStride<SwizzleTileSize, 1, SRMemTileSize,
                                      GSSMemQOStride::swizzle_space()>;
    using RmemShapeQ = RmemShape<N::QO_fragments_per_warp / 2,
                                 SRMemTileFragments / 2, SRMemTilesDHead, 2, 2>;

    using S2RMemLdstConfigQ =
        SRMemLdstConfig<SmemSwizzle, OpS2RSmemStride, OpS2RRmemStride,
                        S2RSmemStrideQ, S2RSmemShapeQ>;

    using S2RSmemShapeKV =
        GSMemShape<B_c, SRMemTileSize, SRMemTilesDHeadPerSwizzleTile,
                   DHeadSwizzleTiles>;

    using S2RSmemStrideK = SMemStride<SwizzleTileSize, 1, SRMemTileSize,
                                      GSSMemKVStride::swizzle_space()>;
    using RmemShapeK = RmemShape<SRMemFragmentsB_c / 1, SRMemTileFragments / 2,
                                 SRMemTilesDHead, 1, 2>;
    using S2RMemLdstConfigK =
        SRMemLdstConfig<SmemSwizzle, OpS2RSmemStride, OpS2RRmemStride,
                        S2RSmemStrideK, S2RSmemShapeKV,
                        true /*SmemRowMajorLdmatrix*/>;

    using S2RSmemStrideV =
        SMemStride<SwizzleTileSize, 1, SRMemTileSize * SwizzleTileSize,
                   GSSMemKVStride::swizzle_space()>;
    using RmemShapeV = RmemShape<SRMemFragmentsDHead / 1,
                                 SRMemTileFragments / 2, SRMemTilesB_c, 1, 2>;
    using S2RMemLdstConfigV =
        SRMemLdstConfig<SmemSwizzle, OpS2RSmemStride, OpS2RRmemStride,
                        S2RSmemStrideV, S2RSmemShapeKV>;

    using R2SSmemShapeO =
        GSMemShape<N::QO_rows_per_warp, RSMemTileSize,
                   RSMemTilesDHeadPerSwizzleTile, DHeadSwizzleTiles>;
    using R2SSmemStrideO = SMemStride<SwizzleTileSize, 1, RSMemTileSize,
                                      GSSMemQOStride::swizzle_space()>;
    using RmemShapeOAccum =
        RmemShape<N::QO_fragments_per_warp / 2,
                  N::d_head_fragments * THR_COLS_PER_ACCUM_FRAGMENT / 2, 1, 2,
                  2>;
    using RmemShapeO =
        RmemShape<N::QO_fragments_per_warp, N::d_head_fragments, 1, 1, 1>;
    using R2SMemLdstConfigO =
        SRMemLdstConfig<SmemSwizzle, OpR2SSMemStride, OpR2SRmemStride,
                        R2SSmemStrideO, R2SSmemShapeO>;

    using RmemShapeSAccum =
        RmemShape<N::QO_fragments_per_warp / 2,
                  N::KV_calc_fragments * THR_COLS_PER_ACCUM_FRAGMENT / 2, 1, 2,
                  2>;
    using RmemShapeP = RmemShape<N::QO_fragments_per_warp / 2,
                                 SRMemTileFragments / 2, SRMemTilesB_c, 2, 2>;

    using RmemConfigQ = RmemLdstConfig<RmemShapeQ, N::Q_rmem_tile_buffer_size,
                                       CFG.Q_mma_load_K_fragments == 0,
                                       false /*RowMajorOpTile*/>;
    using GSRConfigQ = GSRMemLdstConfig<RmemConfigQ, false, N::QO_rows_per_warp,
                                        false /*compute_over_entire_block*/
                                        >;
    using Q_t = GSRBlockTensor<GSRConfigQ, value_t, GSMemLdstConfigQO,
                               S2RMemLdstConfigQ>;

    using RmemConfigK = RmemLdstConfig<RmemShapeK, N::K_rmem_tile_buffer_size,
                                       CFG.K_mma_load_K_fragments == 0,
                                       true /*RowMajorOpTile*/>;
    using GSRConfigK =
        GSRMemLdstConfig<RmemConfigK, false, N::KV_ldst_rows_per_warp,
                         true /*compute_over_entire_block*/
                         >;
    using K_t = GSRBlockTensor<GSRConfigK, value_t, GSMemLdstConfigKV,
                               S2RMemLdstConfigK>;

    using RmemConfigV = RmemLdstConfig<RmemShapeV, N::V_rmem_tile_buffer_size,
                                       CFG.V_mma_load_K_fragments == 0,
                                       true /*RowMajorOpTile*/>;
    using GSRConfigV =
        GSRMemLdstConfig<RmemConfigV, true, N::KV_ldst_rows_per_warp,
                         true /*compute_over_entire_block*/
                         >;
    using V_t = GSRBlockTensor<GSRConfigV, value_t, GSMemLdstConfigKV,
                               S2RMemLdstConfigV>;

    // S/P is kept entirely in the rmem during the entire duration of the
    // kernel.
    using RmemConfigSAccum =
        RmemLdstConfig<RmemShapeSAccum, 1, true /*load_entire_block_into_rf*/,
                       true /*RowMajorOpTile*/>;
    using S_accum_t = RmemBlockTensor<RmemConfigSAccum, accum_t>;
    using RmemConfigP = RmemLdstConfig<RmemShapeP, RmemShapeP::tiles(),
                                       true /*load_entire_block_into_rf*/,
                                       false /*RowMajorOpTile*/>;
    using P_t = RmemBlockTensor<RmemConfigP, value_t>;

    using RmemConfigOAccum =
        RmemLdstConfig<RmemShapeOAccum, 1, true /*load_entire_block_into_rf*/,
                       true /*RowMajorOpTile*/>;
    using O_accum_t = RmemBlockTensor<RmemConfigOAccum, accum_t>;
    using RmemConfigO =
        RmemLdstConfig<RmemShapeO, 1, true /*load_entire_block_into_rf*/,
                       true /*RowMajorOpTile*/>;
    using GSRConfigO = GSRMemLdstConfig<RmemConfigO, false, N::QO_rows_per_warp,
                                        false /*compute_over_entire_block*/
                                        >;
    using O_t = GSRBlockTensor<GSRConfigO, value_t, GSMemLdstConfigQO,
                               R2SMemLdstConfigO>;

    using GEMM_QK = GEMM<Q_t, K_t, S_accum_t, SRMemTilesDHead, value_t>;
    using GEMM_PV = GEMM<P_t, V_t, O_accum_t, SRMemTilesB_c, value_t>;

    using row_statistics_t = ArrayAligned<N::QO_fragments_per_warp, accum_t>;
};

} // namespace flash
// <<< static_kernel_configuration.cuh

namespace flash {

constexpr int debug_warp_rank = 2;
constexpr int debug_block = 0;

using print_cast = float;

FA_DEVICE bool block0() {
    return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
}

FA_DEVICE bool thread0() { return threadIdx.x == 0 && block0(); }

FA_DEVICE bool thread1() { return threadIdx.x == 1 && block0(); }

FA_DEVICE bool is_debug_block() {
    return (blockIdx.x + blockIdx.y * gridDim.x +
            blockIdx.z * gridDim.x * gridDim.y) == debug_block;
}

FA_DEVICE bool is_debug_warp() {
    return is_debug_block() && (threadIdx.x / 32) == debug_warp_rank;
}

FA_DEVICE bool is_warp_leader() {
    return is_debug_warp() && threadIdx.x % 32 == 0;
}

#define printf_leader(fmt, ...)                                                \
    if (is_debug_warp() && is_warp_leader())                                   \
        printf(fmt, ##__VA_ARGS__);

#define printf_warp(fmt, ...)                                                  \
    return;                                                                    \
    if (is_debug_warp())                                                       \
        printf(fmt, ##__VA_ARGS__);

template <typename Tensor, typename castTo = print_cast>
FA_DEVICE void print_smem_matrix(Tensor &t, const char *name = nullptr,
                                 const int iter = -1) {
    __syncthreads();

    using SmemStride = typename Tensor::SmemStride;
    if (thread0()) {
        if (name != nullptr && iter >= 0) {
            printf("%s_%d SMEM:\n", name, iter);
        }

        auto smem_ptr = t.smem_gsmem_ptr;
        for (int row = 0; row < Tensor::GSMemShape::rows(); row++) {
            int cnt = 0;
            for (int stile = 0; stile < Tensor::GSMemShape::swizzle_spaces();
                 stile++) {
                for (int tile = 0; tile < Tensor::GSMemShape::tiles(); tile++) {
                    for (int col = 0; col < Tensor::GSMemShape::cols(); col++) {
                        if constexpr (std::is_same_v<castTo, int>) {
                            printf("%d ", static_cast<castTo>(
                                              smem_ptr[SmemStride::crd2idx(
                                                  row, col, tile, stile)]));
                        } else {
                            printf("%5.2f ", static_cast<castTo>(
                                                 smem_ptr[SmemStride::crd2idx(
                                                     row, col, tile, stile)]));
                        }

                        if (cnt % 8 == 7) {
                            printf(" ");
                        }
                        ++cnt;
                    }
                }
            }
            printf("\n");

            if (row % 8 == 7) {
                printf("\n");
            }
        }
        printf("\n");
    }
    __syncthreads();
}

template <typename Tensor_t_, typename cast_to = print_cast>
FA_DEVICE void print_rmem_matrix(Tensor_t_ &t, const char *name = nullptr,
                                 const int iter = -1,
                                 const int print_tile = -1) {
    const int lane_id = threadIdx.x % 32;
    if (!is_debug_warp()) {
        return;
    }
    if (is_warp_leader() && name != nullptr && iter >= 0) {
        if (print_tile >= 0) {
            printf("%s_%d REGS (print_tile %d):\n", name, iter, print_tile);
        } else {
            printf("%s_%d REGS:\n", name, iter);
        }
    }

    auto view = t.view_with_op_tiling_removed().as_type2();
    using Tensor = decltype(view);

    for (int row_fragment = 0; row_fragment < Tensor::Shape::rows();
         ++row_fragment) {
        if (is_warp_leader()) {
            printf("row: %d\n", row_fragment * 8);
        }
        for (int thr_row = 0; thr_row < 8; ++thr_row) {

            for (int current_tile = 0; current_tile < Tensor::Shape::tiles();
                 ++current_tile) {
                if (print_tile >= 0 && current_tile != print_tile) {
                    continue;
                }

                for (int col_fragment = 0; col_fragment < Tensor::Shape::cols();
                     ++col_fragment) {
                    __syncwarp();

                    for (int thr_col = 0; thr_col < 4; ++thr_col) {
                        int cur_lane = thr_row * 4 + thr_col;
                        if (cur_lane == lane_id) {
                            auto elem =
                                view(row_fragment, col_fragment, current_tile);
                            auto v1 = static_cast<cast_to>(elem.x);
                            auto v2 = static_cast<cast_to>(elem.y);

                            if constexpr (std::is_same_v<cast_to, int>) {
                                printf("%5d %5d ", v1, v2);
                            } else {
                                printf("%5.2f %5.2f ", v1, v2);
                            }
                        }

                        __syncwarp();
                    }
                    if (is_warp_leader()) {
                        printf("  ");
                    }
                }
            }

            if (is_warp_leader()) {
                printf("\n");
            }
        }
    }
    if (is_warp_leader()) {
        printf("\n");
    }
}

template <typename Tensor_t_, typename cast_to = print_cast>
FA_DEVICE void print_rmem_accum_matrix(Tensor_t_ &t, const char *name = nullptr,
                                       const int iter = -1,
                                       const int print_tile = -1) {
    const int lane_id = threadIdx.x % 32;
    if (!is_debug_warp()) {
        return;
    }
    if (is_warp_leader() && name != nullptr && iter >= 0) {
        if (print_tile >= 0) {
            printf("%s_%d REGS (print_tile %d):\n", name, iter, print_tile);
        } else {
            printf("%s_%d REGS:\n", name, iter);
        }
    }

    auto view = t.view_with_op_tiling_removed();
    using Tensor = decltype(view);

    for (int row_fragment = 0; row_fragment < Tensor::Shape::rows();
         ++row_fragment) {
        if (is_warp_leader()) {
            printf("row: %d\n", row_fragment * 8);
        }
        for (int thr_row = 0; thr_row < 8; ++thr_row) {
            for (int current_tile = 0; current_tile < Tensor::Shape::tiles();
                 ++current_tile) {
                if (print_tile >= 0 && current_tile != print_tile) {
                    continue;
                }

                for (int col_fragment = 0; col_fragment < Tensor::Shape::cols();
                     col_fragment += 2) {
                    __syncwarp();

                    for (int thr_col = 0; thr_col < 4; ++thr_col) {
                        int cur_lane = thr_row * 4 + thr_col;
                        if (cur_lane == lane_id) {
                            auto v1 = static_cast<cast_to>(
                                view(row_fragment, col_fragment, current_tile));
                            auto v2 = static_cast<cast_to>(view(
                                row_fragment, col_fragment + 1, current_tile));

                            if constexpr (std::is_same_v<cast_to, int>) {
                                printf("%5d %5d ", v1, v2);
                            } else {
                                printf("%7.2f %7.2f ", v1, v2);
                            }
                        }

                        __syncwarp();
                    }
                    if (is_warp_leader()) {
                        printf("  ");
                    }
                }
            }

            if (is_warp_leader()) {
                printf("\n");
            }
        }
    }
    if (is_warp_leader()) {
        printf("\n");
    }
}

template <typename Array_t, typename cast_to = print_cast>
FA_DEVICE void print_rmem_row(const Array_t &array, const char *name = nullptr,
                              const int iter = -1,
                              bool print_entire_warp = true) {
    const int lane_id = threadIdx.x % 32;
    if (!is_debug_warp()) {
        return;
    }
    if (is_warp_leader() && name != nullptr && iter >= 0) {
        printf("%s_%d REGS:\n", name, iter);
    }

    for (int row_fragment = 0; row_fragment < array.size(); ++row_fragment) {
        for (int t = 0; t < 32; ++t) {
            __syncwarp();
            if (lane_id == t) {
                printf("%5.2f ", static_cast<float>(array[row_fragment]));
            }

            __syncwarp();
            if ((t + 1) % 4 == 0) {
                if (is_warp_leader()) {
                    printf("\n");
                }
            }
        }
        if (is_warp_leader()) {
            printf("\n");
        }
    }
    __syncwarp();
    if (is_warp_leader()) {
        printf("\n\n");
    }
    __syncwarp();
}

// Helper structure to check if a class has op_row and op_col methods
template <typename T, typename = void>
struct has_op_stride : std::false_type {};

template <typename T>
struct has_op_stride<T,
                     std::void_t<decltype(T::op_row()), decltype(T::op_col())>>
    : std::true_type {};

// Helper structure to check if a class has swizzle_space method
template <typename T, typename = void>
struct has_swizzle_tile : std::false_type {};

template <typename T>
struct has_swizzle_tile<T, std::void_t<decltype(T::swizzle_space())>>
    : std::true_type {};

// Helper structure to check if a class has op_rows and op_cols methods
template <typename T, typename = void>
struct has_op_shape : std::false_type {};

template <typename T>
struct has_op_shape<T,
                    std::void_t<decltype(T::op_rows()), decltype(T::op_cols())>>
    : std::true_type {};

// Helper structure to check if a class has swizzle_spaces method
template <typename T, typename = void>
struct has_swizzle_tiles : std::false_type {};

template <typename T>
struct has_swizzle_tiles<T, std::void_t<decltype(T::swizzle_spaces())>>
    : std::true_type {};

// Helper function to print stride information
template <typename Stride>
FA_DEVICE static void print_stride(const char *name) {
    printf("  %s: {row: %d, col: %d, tile: %d", name, Stride::row(),
           Stride::col(), Stride::tile());

    // Print op_row and op_col if they exist
    if constexpr (has_op_stride<Stride>::value) {
        printf(", op_row: %d, op_col: %d", Stride::op_row(), Stride::op_col());
    }

    // Print swizzle_space if it exists
    if constexpr (has_swizzle_tile<Stride>::value) {
        printf(", swizzle_space: %d", Stride::swizzle_space());
    }

    printf("}\n");
}

// Helper function to print shape information
template <typename Shape>
FA_DEVICE static void print_shape(const char *name) {
    printf("  %s: {rows: %d, cols: %d, tiles: %d", name, Shape::rows(),
           Shape::cols(), Shape::tiles());

    // Print op_rows and op_cols if they exist
    if constexpr (has_op_shape<Shape>::value) {
        printf(", op_rows: %d, op_cols: %d", Shape::op_rows(),
               Shape::op_cols());
    }

    // Print swizzle_spaces if it exists
    if constexpr (has_swizzle_tiles<Shape>::value) {
        printf(", swizzle_spaces: %d", Shape::swizzle_spaces());
    }

    printf("}\n");
}

// Helper function to print tensor type information
template <typename Tensor>
FA_DEVICE static void print_tensor_type(const char *name) {
    printf("\n%s:\n", name);
    // print_shape<typename Tensor::StorageShape>("StorageShape");
    print_shape<typename Tensor::Shape>("Shape");
    print_stride<typename Tensor::Stride>("Stride");
    printf("  StorageSize: %d\n", Tensor::StorageSize);
    printf("  rmem_tile_buffer_size: %d\n", Tensor::rmem_tile_buffer_size);
    printf("  load_entire_block_into_rf: %d\n",
           Tensor::load_entire_block_into_rf);

    // Call the layout print function to show detailed layout information
    printf("\n  Layout Details for %s:\n", name);
    using Layout = typename Tensor::Layout;
    Layout::print();
}

// Helper function to print OpIters information for GSMemLdstConfig
template <typename GSMemCfg>
FA_DEVICE static void print_gsmem_opiters(const char *name) {
    printf("\n  %s OpIters:\n", name);
    printf("    rows: %d, cols: %d\n", GSMemCfg::OpIters::rows(),
           GSMemCfg::OpIters::cols());
    printf("    swizzle_spaces: %d\n", GSMemCfg::OpIters::swizzle_spaces());
}

// Helper function to print OpIters information for SRMemLdstConfig
template <typename SRMemCfg>
FA_DEVICE static void print_srmem_opiters(const char *name) {
    printf("\n  %s OpIters:\n", name);
    printf("    rows: %d, cols: %d\n", SRMemCfg::OpIters::rows(),
           SRMemCfg::OpIters::cols());
    printf("    tiles: %d, swizzle_spaces: %d\n", SRMemCfg::OpIters::tiles(),
           SRMemCfg::OpIters::swizzle_spaces());
}

// Print configuration as a static member function of FlashKernelTypes
template <typename Kernel>
FA_DEVICE static void print_config() {
    using N = typename Kernel::N;
    printf("\nFlashKernelTypes Configuration:\n");
    printf("----------------------------------------\n");

    // Print Kernel Configuration
    printf("\nKernel Configuration:\n");
    printf("  B_r: %d\n", Kernel::B_r);
    printf("  B_c: %d\n", Kernel::B_c);
    printf("  d_head: %d\n", Kernel::d_head);
    printf("  n_warps: %d\n", Kernel::n_warps);
    printf("  swizzled: %d\n", Kernel::swizzled);
    printf("  async_copy: %d\n", Kernel::async_copy);
    printf("  eager_load_blocks: %d\n", Kernel::eager_load_blocks);
    printf("  optimized_softmax: %d\n", Kernel::optimized_softmax);
    printf("  mma_double_buffer_loads: %d\n",
           Kernel::CFG.mma_double_buffer_loads);
    printf("  Q_mma_load_K_fragments: %d\n",
           Kernel::CFG.Q_mma_load_K_fragments);
    printf("  K_mma_load_K_fragments: %d\n",
           Kernel::CFG.K_mma_load_K_fragments);
    printf("  V_mma_load_K_fragments: %d\n",
           Kernel::CFG.V_mma_load_K_fragments);

    // Print Tile Configurations
    printf("\nTile Configurations:\n");
    printf("  n_threads: %d\n", N::n_threads);
    printf("  d_head_fragments: %d\n", N::d_head_fragments);
    printf("  QO_rows_per_warp: %d\n", N::QO_rows_per_warp);
    printf("  QO_fragments_per_warp: %d\n", N::QO_fragments_per_warp);
    printf("  KV_calc_fragments: %d\n", N::KV_calc_fragments);
    printf("  KV_ldst_fragments_per_warp: %d\n", N::KV_ldst_fragments_per_warp);
    printf("  KV_ldst_rows_per_warp: %d\n", N::KV_ldst_rows_per_warp);
    printf("  QK_rmem_tile_fragments: %d\n", N::QK_rmem_tile_fragments);
    printf("  QK_rmem_tile_size: %d\n", N::QK_rmem_tile_size);
    printf("  QK_rmem_tiles: %d\n", N::QK_rmem_tiles);
    printf("  PV_rmem_tile_fragments: %d\n", N::PV_rmem_tile_fragments);
    printf("  PV_rmem_tile_size: %d\n", N::PV_rmem_tile_size);
    printf("  PV_rmem_tiles: %d\n", N::PV_rmem_tiles);
    printf("  Q_rmem_tile_buffer_size: %d\n", N::Q_rmem_tile_buffer_size);
    printf("  K_rmem_tile_buffer_size: %d\n", N::K_rmem_tile_buffer_size);
    printf("  V_rmem_tile_buffer_size: %d\n", N::V_rmem_tile_buffer_size);

    // Print dimensions & constants
    printf("\nDimensions & Constants:\n");
    printf("  SwizzleTileSize: %d\n", Kernel::SwizzleTileSize);
    printf("  DHeadSwizzleTiles: %d\n", Kernel::DHeadSwizzleTiles);
    printf("  B_c_SwizzleTiles: %d\n", Kernel::B_c_SwizzleTiles);
    printf("  SRMemTileSize: %d\n", Kernel::SRMemTileSize);
    printf("  SRMemTileFragments: %d\n", Kernel::SRMemTileFragments);
    printf("  SRMemTilesDHead: %d\n", Kernel::SRMemTilesDHead);
    printf("  RSMemTileSize: %d\n", Kernel::RSMemTileSize);
    printf("  RSMemTilesDHead: %d\n", Kernel::RSMemTilesDHead);

    // Print Strides
    printf("\nBase Strides:\n");
    print_stride<typename Kernel::OpGSMemStride>("OpGSMemStride");
    print_stride<typename Kernel::OpS2RSmemStride>("OpS2RSmemStride");
    print_stride<typename Kernel::OpS2RRmemStride>("OpS2RRmemStride");
    print_stride<typename Kernel::OpR2SSMemStride>("OpR2SSMemStride");
    print_stride<typename Kernel::OpR2SRmemStride>("OpR2SRmemStride");

    // Print GSSMemStrides
    printf("\nGSSMem Strides:\n");
    print_stride<typename Kernel::GSSMemQOStride>("GSSMemQOStride");
    print_stride<typename Kernel::GSSMemKVStride>("GSSMemKVStride");

    // Print OpIters for GSMemLdstConfig types
    printf("\nGSMemLdstConfig OpIters:\n");
    print_gsmem_opiters<typename Kernel::GSMemLdstConfigQO>(
        "GSMemLdstConfigQO");
    print_gsmem_opiters<typename Kernel::GSMemLdstConfigKV>(
        "GSMemLdstConfigKV");

    // Print OpIters for SRMemLdstConfig types
    printf("\nSRMemLdstConfig OpIters:\n");
    print_srmem_opiters<typename Kernel::S2RMemLdstConfigQ>(
        "S2RMemLdstConfigQ");
    print_srmem_opiters<typename Kernel::S2RMemLdstConfigK>(
        "S2RMemLdstConfigK");
    print_srmem_opiters<typename Kernel::S2RMemLdstConfigV>(
        "S2RMemLdstConfigV");
    print_srmem_opiters<typename Kernel::R2SMemLdstConfigO>(
        "R2SMemLdstConfigO");

    // Print Shapes and Memory Configurations
    printf("\nShapes and Memory:\n");
    print_shape<typename Kernel::GSMemShapeQO>("GSMemShapeQO");
    print_shape<typename Kernel::GSMemShapeKV>("GSMemShapeKV");

    printf("\nS2R Memory Shapes:\n");
    print_shape<typename Kernel::S2RSmemShapeQ>("S2RSmemShapeQ");

    // Note: Skip shapes that might not be defined in all configurations

    printf("\nRmem Shapes:\n");
    print_shape<typename Kernel::RmemShapeQ>("RmemShapeQ");
    print_shape<typename Kernel::RmemShapeK>("RmemShapeK");
    print_shape<typename Kernel::RmemShapeV>("RmemShapeV");
    print_shape<typename Kernel::RmemShapeO>("RmemShapeO");
    print_shape<typename Kernel::RmemShapeOAccum>("RmemShapeOAccum");
    print_shape<typename Kernel::RmemShapeSAccum>("RmemShapeSAccum");
    print_shape<typename Kernel::RmemShapeP>("RmemShapeP");

    printf("\nSpecific Strides:\n");
    print_stride<typename Kernel::S2RSmemStrideQ>("S2RSmemStrideQ");
    print_stride<typename Kernel::S2RSmemStrideK>("S2RSmemStrideK");
    print_stride<typename Kernel::S2RSmemStrideV>("S2RSmemStrideV");
    print_stride<typename Kernel::R2SSmemStrideO>("R2SSmemStrideO");

    printf("\nRmemMatrix Configurations:\n");
    print_tensor_type<typename Kernel::O_accum_t>("O_accum_t");
    print_tensor_type<typename Kernel::S_accum_t>("S_accum_t");
    print_tensor_type<typename Kernel::P_t>("P_t");

    printf("\nTensor Types:\n");
    print_tensor_type<typename Kernel::Q_t>("Q_t (Base RmemTensor)");
    print_tensor_type<typename Kernel::K_t>("K_t (Base RmemTensor)");
    print_tensor_type<typename Kernel::V_t>("V_t (Base RmemTensor)");
    print_tensor_type<typename Kernel::O_t>("O_t (Base RmemTensor)");

    // Print GEMM Configurations
    printf("\nGEMM Configurations:\n");

    printf("GEMM_QK:\n");
    printf("  Tiles: %d\n", Kernel::GEMM_QK::Tiles);
    printf("  DoubleBufferA: %d\n", Kernel::GEMM_QK::DoubleBufferA);
    printf("  DoubleBufferB: %d\n", Kernel::GEMM_QK::DoubleBufferB);
    printf("  DoubleBuffer: %d\n", Kernel::GEMM_QK::DoubleBuffer);

    printf("\nGEMM_PV:\n");
    printf("  Tiles: %d\n", Kernel::GEMM_PV::Tiles);
    printf("  DoubleBufferA: %d\n", Kernel::GEMM_PV::DoubleBufferA);
    printf("  DoubleBufferB: %d\n", Kernel::GEMM_PV::DoubleBufferB);
    printf("  DoubleBuffer: %d\n", Kernel::GEMM_PV::DoubleBuffer);

    printf("----------------------------------------\n");
}

} // namespace flash
// <<< debug.cuh
// >>> flash_attention.cuh
// <<< flash_attention.cuh
// >>> gemm.cuh
// <<< gemm.cuh
// >>> ptx_functions.cuh
// <<< ptx_functions.cuh
// >>> softmax.cuh

// >>> array.cuh
// <<< array.cuh
// >>> common.h
// <<< common.h

namespace flash {

/*
Each group of 4 threads contains a row.

*/

template <bool is_first, typename S_accum_t, typename RowT,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void calc_row_max(S_accum_t &S_accum, RowT &m) {
    FA_UNROLL
    for (int q = 0; q < S_accum_t::Shape::rows(); ++q) {
        if constexpr (is_first) {
            m[q] = S_accum(q, 0);
        } else {
            m[q] = max(m[q], S_accum(q, 0));
        }

        // Calculate max for row across all in-thread registers.
        FA_UNROLL
        for (int k = 1; k < S_accum_t::Shape::cols(); ++k) {
            m[q] = max(m[q], S_accum(q, k));
        }

        // Group reduction
        m[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m[q], 2), m[q]);
        m[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m[q], 1), m[q]);
    }
}

template <typename O_accum_t, typename RowT, typename accum_t = float>
FA_DEVICE_CONSTEXPR void scale_l_O(RowT &m_cur, RowT &m_prev, RowT &l,
                                   O_accum_t &O_accum,
                                   const accum_t &softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < O_accum_t::Shape::rows(); ++q) {
        accum_t scale = exp2f((m_prev[q] - m_cur[q]) * softmax_scale);
        l[q] *= scale;
        FA_UNROLL
        for (int d_head = 0; d_head < O_accum_t::Shape::cols(); ++d_head) {
            O_accum(q, d_head) *= scale;
        }
    }
}

template <typename S_accum_t, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
exponentiate_tensor(S_accum_t &S_accum,
                    ArrayAligned<S_accum_t::Shape::rows(), accum_t> &m,
                    const accum_t &softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < S_accum_t::Shape::rows(); ++q) {
        accum_t max_scaled = m[q] * softmax_scale;
        FA_UNROLL
        for (int k = 0; k < S_accum_t::Shape::cols(); ++k) {
            S_accum(q, k) = exp2f(S_accum(q, k) * softmax_scale - max_scaled);
        }
    }
}

template <bool is_first, typename P_accum_t, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
update_row_exp_sum(P_accum_t &P_accum,
                   ArrayAligned<P_accum_t::Shape::rows(), accum_t> &l) {
    FA_UNROLL
    for (int q = 0; q < P_accum_t::Shape::rows(); ++q) {
        if constexpr (is_first) {
            l[q] = P_accum(q, 0);
        } else {
            l[q] += P_accum(q, 0);
        }

        FA_UNROLL
        for (int d_head = 1; d_head < P_accum_t::Shape::cols(); ++d_head) {
            l[q] += P_accum(q, d_head);
        }
    }
}

template <bool is_first, bool optimized_softmax, typename S_accum_untiled_t,
          typename O_accum_untiled_t, typename row_statistics_t,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void local_softmax(S_accum_untiled_t &S_accum_untiled,
                                       O_accum_untiled_t &O_accum_untiled,
                                       row_statistics_t &m, row_statistics_t &l,
                                       const accum_t &softmax_scale) {
    if constexpr (is_first && optimized_softmax) {
        calc_row_max<is_first>(S_accum_untiled, m);
        exponentiate_tensor(S_accum_untiled, m, softmax_scale);
        update_row_exp_sum<is_first>(S_accum_untiled, l);
    } else {
        row_statistics_t m_prev;
        m_prev.copy(m);
        calc_row_max<is_first>(S_accum_untiled, m);

        scale_l_O(m, m_prev, l, O_accum_untiled, softmax_scale);
        exponentiate_tensor(S_accum_untiled, m, softmax_scale);
        update_row_exp_sum<is_first>(S_accum_untiled, l);
    }
}

template <typename O_accum_untiled_t, typename row_statistics_t,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
final_softmax_normalization(O_accum_untiled_t &O_accum_untiled,
                            row_statistics_t &l) {
    // Finish reduction sum across all threads in the same row.
    FA_UNROLL
    for (int q = 0; q < row_statistics_t::size(); ++q) {
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
        l[q] = 1.0f / l[q];
    }

    FA_UNROLL
    for (int q = 0; q < O_accum_untiled_t::Shape::rows(); ++q) {
        FA_UNROLL
        for (int d_head = 0; d_head < O_accum_untiled_t::Shape::cols();
             ++d_head) {
            O_accum_untiled(q, d_head) *= l[q];
        }
    }
}

} // namespace flash// <<< softmax.cuh
// >>> static_kernel_configuration.cuh
// <<< static_kernel_configuration.cuh
// >>> tensor.cuh
// <<< tensor.cuh

namespace flash {

template <bool is_first, bool optimized_softmax, typename Q_t, typename K_t,
          typename V_t, typename S_accum_t, typename P_t, typename O_accum_t,
          typename row_statistics_t, typename GEMM_QK, typename GEMM_PV>
FA_DEVICE void process_kv_block(Q_t &Q, K_t &K, V_t &V, O_accum_t &O_accum,
                                row_statistics_t &m, row_statistics_t &l,
                                const float &softmax_scale, const int &block) {

    S_accum_t S_accum;
    // Initialize the registers for S to 0.
    S_accum.zero();

    // Block until we've copied the K block-tile for this iteration into
    // shared memory.
    cp_async_wait<0>();
    // After this barrier, it is safe to load the next V block, because all
    // warps have done the previous PV matmul.
    __syncthreads();

    // Start the (async) copy for the V matrix from gmem to smem but
    // do not wait until after the S=QK matmul.
    V.copy_GM2SM(block);
    cp_async_commit();

    if constexpr (K_t::load_entire_block_into_rf) {
        K.copy_SM2RF_all_tiles();
    }

    matmul<GEMM_QK>(Q, K, S_accum);

    // Wait for V to finish loading.
    cp_async_wait<0>();
    // After this barrier, it is safe to load the next block of K.
    __syncthreads();

    if constexpr (is_first) {
        // Initialize the accumulator for O.
        O_accum.zero();
    }

    // Start the async copy for the next K block-tile from gmem to
    // smem, but do not wait for the copy until the next iteration
    // when we need it.
    if (block > 0) {
        K.copy_GM2SM(block - 1);
        cp_async_commit();
    }

    // Online softmax
    auto S_accum_untiled = S_accum.view_with_op_tiling_removed();
    auto O_accum_no_op_tiling = O_accum.view_with_op_tiling_removed();
    local_softmax<is_first, optimized_softmax>(
        S_accum_untiled, O_accum_no_op_tiling, m, l, softmax_scale);

    P_t P_b16;
    auto S_accum_view = S_accum.view();
    auto P_b16_view = P_b16.view();
    // Convert the S accumulator block into P fp16 input block.
    convert_to_16_bit_dtype(S_accum_view, P_b16_view);

    if constexpr (V_t::load_entire_block_into_rf) {
        V.copy_SM2RF_all_tiles();
    }

    matmul<GEMM_PV>(P_b16, V, O_accum);
}

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
    using P_t = typename Kernel::P_t;
    using O_t = typename Kernel::O_t;
    using S_accum_t = typename Kernel::S_accum_t;
    using O_accum_t = typename Kernel::O_accum_t;
    using GEMM_QK = typename Kernel::GEMM_QK;
    using GEMM_PV = typename Kernel::GEMM_PV;
    using row_statistics_t = typename Kernel::row_statistics_t;

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

    extern __shared__ __align__(16) char smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(smem);
    value_t *smem_O = smem_Q;
    value_t *smem_K = &smem_Q[Kernel::B_r * Kernel::d_head];
    value_t *smem_V = &smem_K[Kernel::B_c * Kernel::d_head];

    // Pointers to the K&V locations in smem that the warp copies to.
    Q_t Q(gmem_Q, smem_Q);
    K_t K(gmem_K, smem_K);
    V_t V(gmem_V, smem_V);

    // The accumulator for O is only kept in registers. At the end of the
    // kernel, it is then converted into a 16-bit type and then copied into
    // gmem.
    O_accum_t O_accum;
    auto O_accum_no_op_tiling = O_accum.view_with_op_tiling_removed();

    int block = args.n_KV_blocks - 1;
    // Start the async copy of the Q and K tiles.
    Q.copy_GM2SM(0);
    cp_async_commit();
    K.copy_GM2SM(block);
    cp_async_commit();

    // Initialize softmax_scale, m, and l.
    const accum_t softmax_scale =
        rsqrt(static_cast<accum_t>(Kernel::d_head)) * M_LOG2E;
    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();

    // Replace raw arrays with Array objects
    row_statistics_t m;
    row_statistics_t l;

    if constexpr (!Kernel::optimized_softmax) {
        m.fill(neg_inf);
        l.fill(0.0);
    }

    if constexpr (Q_t::load_entire_block_into_rf) {
        // We only wait for the Q block to finish loading.
        cp_async_wait<1>();

        // We need the __syncthreads() in addition to the cp_async_wait()
        // because cp_async_wait() only blocks until the current thread has
        // finished loading. The entire CTA will read this block from
        // smem, so we need to wait on a CTA-wide barrier.
        __syncthreads();
        Q.copy_SM2RF_all_tiles();
    }

    process_kv_block<true, Kernel::optimized_softmax, Q_t, K_t, V_t, S_accum_t,
                     P_t, O_accum_t, row_statistics_t, GEMM_QK, GEMM_PV>(
        Q, K, V, O_accum, m, l, softmax_scale, block);

    --block;
    for (; block >= 0; --block) {
        process_kv_block<false, Kernel::optimized_softmax, Q_t, K_t, V_t,
                         S_accum_t, P_t, O_accum_t, row_statistics_t, GEMM_QK,
                         GEMM_PV>(Q, K, V, O_accum, m, l, softmax_scale, block);
    }

    final_softmax_normalization(O_accum_no_op_tiling, l);

    O_t O_b16(gmem_O, smem_O);
    auto O_b16_view = O_b16.view();
    convert_to_16_bit_dtype(O_accum_no_op_tiling, O_b16_view);

    // Instead of writing directly to gmem, we write to smem as an intermediary
    // step. This allows us to
    // - use 16B vectorized stores, as opposed to 4B stores
    // - fully coalesce our stores
    //   - each warp can store 4x128B aligned lines (512B/warp) instead
    //   of 8x16B uncoalesced rows (128B/warp)
    O_b16.copy_RF2SM();

    // Wait until all threads have written to smem.
    __syncthreads();

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
    float softmax_scale = M_LOG2E / sqrtf(d_head);

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