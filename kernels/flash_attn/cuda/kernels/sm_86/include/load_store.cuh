#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "common.cuh"
#include "ptx_functions.cuh"
#include "swizzling.cuh"

// =============================================================================
// load_store.cuh — tile copy functions between gmem, smem, and registers
//
// KEY SIMPLIFICATION vs src_1-7:
//   The original code used C++20 non-type template parameters (NTTPs) of struct
//   type (TensorLDSTConfig, LDSTCommon, TileLayout).  Those are opaque to most
//   CUDA tutorials and require C++20.
//
//   Here every function is templated on plain int / bool parameters:
//     WARP_ROWS   — rows this warp handles in one gmem<->smem copy
//     SMEM_COLS   — element stride of the smem tile (= d_head)
//     SWIZZLED    — apply XOR swizzle to smem addresses
//     ASYNC_COPY  — use cp.async instead of synchronous uint4 loads
//     RF_ROWS     — row fragment count in the register file
//     RF_COLS     — col fragment count per MMA step in the register file
//   All of these are constexpr ints/bools, so the compiler sees them as
//   compile-time constants and can fully unroll loops.
// =============================================================================

namespace flash {

// ---------------------------------------------------------------------------
// gmem -> smem copy (WARP_ROWS rows of a tile)
//
// Each call copies a (WARP_ROWS, SMEM_COLS) sub-tile owned by one warp.
// Inner loop: 4 rows × full width per iteration, using 128-bit accesses.
// ---------------------------------------------------------------------------
template <int WARP_ROW_FRAGS,   // # 8-row fragments this warp copies
          int SMEM_COLS,        // smem row stride in elements (= d_head)
          bool SWIZZLED,        // XOR swizzle smem addresses?
          bool ASYNC_COPY,      // use cp.async (non-blocking)?
          typename T>
FA_DEVICE_CONSTEXPR void
copy_tile_g2s(T *gmem, T *smem, int64_t gmem_stride, int lane_id) {

    // Number of 4-row passes this warp needs
    constexpr int N_ROW_ITERS   = WARP_ROW_FRAGS * ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER;
    // Threads per row pass (= warp / 4 rows = 32/4 = 8)
    constexpr int COLS_PER_ITER = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;
    // Total 8-element column groups in smem
    constexpr int N_COL_GROUPS  = SMEM_COLS / COLS_PER_FRAGMENT;

    const int r_thread = lane_id / COLS_PER_ITER;   // row within the 4-row pass
    const int c_thread = lane_id % COLS_PER_ITER;   // column group assigned to this thread

    FA_UNROLL
    for (int ri = 0; ri < N_ROW_ITERS; ++ri) {
        const int row = ri * GSM_LDST_ROWS_PER_ITER + r_thread;
        FA_UNROLL
        for (int c = 0; c < N_COL_GROUPS; c += COLS_PER_ITER) {
            const int gmem_col = c + c_thread;
            const int smem_col = get_smem_col_fragment<N_COL_GROUPS, SWIZZLED>(row, gmem_col);

            T *dst = &smem[row * SMEM_COLS + smem_col * COLS_PER_FRAGMENT];
            T *src = &gmem[row * gmem_stride + gmem_col * COLS_PER_FRAGMENT];

            if constexpr (ASYNC_COPY)
                cp_async<BYTES_PER_VEC4_ACCESS>(dst, src);
            else
                reinterpret_cast<uint4 *>(dst)[0] = reinterpret_cast<uint4 *>(src)[0];
        }
    }
}

// ---------------------------------------------------------------------------
// smem -> gmem copy (WARP_ROWS rows of a tile)
// Same layout as copy_tile_g2s but direction is reversed.
// ---------------------------------------------------------------------------
template <int WARP_ROW_FRAGS,
          int SMEM_COLS,
          bool SWIZZLED,
          typename T>
FA_DEVICE_CONSTEXPR void
copy_tile_s2g(T *gmem, T *smem, int64_t gmem_stride, int lane_id) {

    constexpr int N_ROW_ITERS   = WARP_ROW_FRAGS * ROWS_PER_FRAGMENT / GSM_LDST_ROWS_PER_ITER;
    constexpr int COLS_PER_ITER = WARP_SIZE / GSM_LDST_ROWS_PER_ITER;
    constexpr int N_COL_GROUPS  = SMEM_COLS / COLS_PER_FRAGMENT;

    const int r_thread = lane_id / COLS_PER_ITER;
    const int c_thread = lane_id % COLS_PER_ITER;

    FA_UNROLL
    for (int ri = 0; ri < N_ROW_ITERS; ++ri) {
        const int row = ri * GSM_LDST_ROWS_PER_ITER + r_thread;
        FA_UNROLL
        for (int c = 0; c < N_COL_GROUPS; c += COLS_PER_ITER) {
            const int gmem_col = c + c_thread;
            const int smem_col = get_smem_col_fragment<N_COL_GROUPS, SWIZZLED>(row, gmem_col);

            reinterpret_cast<uint4 *>(&gmem[row * gmem_stride + gmem_col * COLS_PER_FRAGMENT])[0] =
                reinterpret_cast<uint4 *>(&smem[row * SMEM_COLS + smem_col * COLS_PER_FRAGMENT])[0];
        }
    }
}

// ---------------------------------------------------------------------------
// smem -> registers (non-transposed): loads Q or K fragments
//
// Each ldmatrix.x4 loads a (16×16) chunk = 2 row-frags × 2 col-frags.
// RF_ROWS × RF_COLS register tiles are filled.
// col_offset lets you start loading at a different K-tile (used in
// streaming / double-buffer mode).
// ---------------------------------------------------------------------------
template <int RF_ROWS,    // row fragments in register file
          int RF_COLS,    // col fragments per MMA step
          int SMEM_COLS,  // smem row stride in elements
          bool SWIZZLED,
          typename T>
FA_DEVICE_CONSTEXPR void
load_smem2rf(uint32_t (&regs)[RF_ROWS][RF_COLS],
             T *smem, int lane_id, int col_offset = 0) {

    constexpr int ROW_FRAGS_PER_ITER = 2;   // ldmatrix.x4 covers 2 row-frags at once
    constexpr int ROWS_PER_ITER  = ROWS_PER_FRAGMENT * ROW_FRAGS_PER_ITER;  // 16
    constexpr int TOTAL_COL_FRAGS = SMEM_COLS / ELEMS_PER_VEC4_ACCESS;      // d_head / 8
    constexpr int COLS_PER_ITER  = WARP_SIZE / ROWS_PER_ITER;               // 32 / 16 = 2

    const int r_thread = lane_id % ROWS_PER_ITER;
    const int c_thread = lane_id / ROWS_PER_ITER;

    FA_UNROLL
    for (int r = 0; r < RF_ROWS; r += ROW_FRAGS_PER_ITER) {
        const int row = r_thread + r * ROWS_PER_FRAGMENT;
        FA_UNROLL
        for (int c = 0; c < RF_COLS; c += COLS_PER_ITER) {
            const int col = c_thread + c + col_offset;
            const int smem_col = get_smem_col_fragment<TOTAL_COL_FRAGS, SWIZZLED>(row, col);
            ldmatrix_x4(
                &smem[row * SMEM_COLS + smem_col * ELEMS_PER_VEC4_ACCESS],
                regs[r][c],       regs[r + 1][c],
                regs[r][c + 1],   regs[r + 1][c + 1]);
        }
    }
}

// ---------------------------------------------------------------------------
// smem -> registers (transposed): loads V fragments
//
// Uses ldmatrix.x4.trans so the register layout is transposed vs smem.
// row_offset lets you stream-load one K-slice of V at a time.
// ---------------------------------------------------------------------------
template <int RF_ROWS,
          int RF_COLS,
          int SMEM_COLS,
          bool SWIZZLED,
          typename T>
FA_DEVICE_CONSTEXPR void
load_smem2rf_T(uint32_t (&regs)[RF_ROWS][RF_COLS],
               T *smem, int lane_id, int row_offset = 0) {

    constexpr int ROW_FRAGS_PER_ITER = 2;
    constexpr int ROWS_PER_ITER  = ROWS_PER_FRAGMENT * ROW_FRAGS_PER_ITER;
    constexpr int TOTAL_COL_FRAGS = SMEM_COLS / ELEMS_PER_VEC4_ACCESS;
    constexpr int COLS_PER_ITER  = WARP_SIZE / ROWS_PER_ITER;

    const int r_thread = lane_id % ROWS_PER_ITER;
    const int c_thread = lane_id / ROWS_PER_ITER;

    // For the transposed case the outer loop iterates over RF_COLS (K dimension)
    // and the inner over RF_ROWS (output N dimension).
    FA_UNROLL
    for (int r = 0; r < RF_COLS; r += ROW_FRAGS_PER_ITER) {
        const int row = r_thread + (r + row_offset) * ROWS_PER_FRAGMENT;
        FA_UNROLL
        for (int c = 0; c < RF_ROWS; c += COLS_PER_ITER) {
            const int smem_col = get_smem_col_fragment<TOTAL_COL_FRAGS, SWIZZLED>(row, c_thread + c);
            ldmatrix_x4_transpose(
                &smem[row * SMEM_COLS + smem_col * ELEMS_PER_VEC4_ACCESS],
                regs[c][r],       regs[c][r + 1],
                regs[c + 1][r],   regs[c + 1][r + 1]);
        }
    }
}

// ---------------------------------------------------------------------------
// registers -> smem: stores O fragments back for vectorized gmem write
//
// Each thread writes 2 fp16 elements per fragment.
// After this call a __syncwarp() is needed before the smem->gmem copy.
// ---------------------------------------------------------------------------
template <int RF_ROWS,
          int RF_COLS,
          int SMEM_COLS,
          bool SWIZZLED,
          typename T>
FA_DEVICE_CONSTEXPR void
store_rf2smem(uint32_t (&regs)[RF_ROWS][RF_COLS],
              T *smem, int lane_id) {

    constexpr int TOTAL_COL_FRAGS  = SMEM_COLS / ELEMS_PER_VEC4_ACCESS;
    constexpr int ELEMS_PER_STORE  = 2;         // one uint32 holds 2 fp16 values

    const int r_thread  = lane_id / 4;
    const int c_inner   = (lane_id % 4) * ELEMS_PER_STORE;

    FA_UNROLL
    for (int r = 0; r < RF_ROWS; ++r) {
        const int row = r_thread + r * ROWS_PER_FRAGMENT;
        FA_UNROLL
        for (int c = 0; c < RF_COLS; ++c) {
            const int smem_col = get_smem_col_fragment<TOTAL_COL_FRAGS, SWIZZLED>(row, c);
            reinterpret_cast<uint32_t *>(
                &smem[row * SMEM_COLS + smem_col * ELEMS_PER_VEC4_ACCESS + c_inner]
            )[0] = regs[r][c];
        }
    }
}

// ---------------------------------------------------------------------------
// Type-conversion: float32 accumulator -> fp16 / bf16 register tile
//
// Used to convert S_accum (float) -> P_b16 (half/bf16) before the PV matmul,
// and to convert O_accum (float) -> O_b16 for the final smem write.
// ---------------------------------------------------------------------------
template <typename value_t, int M_fragments, int N_fragments>
FA_DEVICE_CONSTEXPR void
convert_to_16_bit_dtype(float (&src_float)[M_fragments][N_fragments * 2],
                        uint32_t (&dest_uint)[M_fragments][N_fragments]) {
    using value2_t =
        std::conditional_t<std::is_same_v<value_t, half>, half2, nv_bfloat162>;

    float2 (&src)[M_fragments][N_fragments] =
        reinterpret_cast<float2 (&)[M_fragments][N_fragments]>(src_float);
    value2_t (&dest)[M_fragments][N_fragments] =
        reinterpret_cast<value2_t (&)[M_fragments][N_fragments]>(dest_uint);

    FA_UNROLL
    for (int m = 0; m < M_fragments; ++m) {
        FA_UNROLL
        for (int n = 0; n < N_fragments; ++n) {
            if constexpr (std::is_same_v<value_t, half>)
                dest[m][n] = __float22half2_rn(src[m][n]);
            else
                dest[m][n] = __float22bfloat162_rn(src[m][n]);
        }
    }
}

} // namespace flash
