#pragma once

#include "common.cuh"
#include "load_store.cuh"

// =============================================================================
// tensor.cuh — register-file tile types
//
// KEY SIMPLIFICATION vs src_1-7:
//   The original MatrixLDST<TensorLDSTConfig, value_t> took a struct as a
//   non-type template parameter (C++20).  Here we expose every dimension as a
//   plain int / bool template parameter so the code reads like standard CUDA.
//
// Three tile types are provided:
//
//   TileBuffer<...> — fp16/bf16 tile with full gmem <-> smem <-> RF pipeline.
//                     Used for Q, K, V, O_value.
//
//   AccumTile<ROWS, COLS>  — float32 accumulator (RF only).
//                            Used for S_accum and O_accum.
//
//   RFTile<ROWS, COLS, T>  — fp16/bf16 RF-only tile (no gmem/smem ops).
//                            Used for P_b16 (fp16 copy of S before PV matmul).
// =============================================================================

namespace flash {

// ---------------------------------------------------------------------------
// TileBuffer
//
// Template parameters (all plain int/bool — no struct NTTPs):
//   GSM_ROWS     — rows this warp copies between gmem and smem
//                  (= GSM.row_fragments * ROWS_PER_FRAGMENT in src_1-7)
//   RF_ROW_FRAGS — row fragment count in the register file
//   RF_COL_FRAGS — col fragment count per MMA step
//   SMEM_COLS    — element stride of the smem tile (= d_head)
//   BLOCK_ROWS   — total tile row count (B_r or B_c), used to advance gmem ptr
//   WHOLE_BLOCK  — true  → read smem from the tile base (full K/V block);
//                  false → read smem from this warp's own chunk (Q/O)
//   WHOLE_RF     — true  → entire block already in RF; false → stream per step
//   MMA_STAGES   — 1 = single buffer, 2 = double-buffer
//   TRANSPOSED   — use transposed ldmatrix (V matrix)
//   ASYNC_CP     — use cp.async for gmem→smem
//   SWIZZLED     — XOR swizzle smem addresses
//   T            — half or nv_bfloat16
// ---------------------------------------------------------------------------
template <
    int  GSM_ROWS,
    int  RF_ROW_FRAGS,
    int  RF_COL_FRAGS,
    int  SMEM_COLS,
    int  BLOCK_ROWS,
    bool WHOLE_BLOCK,
    bool WHOLE_RF,
    int  MMA_STAGES,
    bool TRANSPOSED,
    bool ASYNC_CP,
    bool SWIZZLED,
    typename T
>
struct TileBuffer {
    // Register storage: [pipeline_stage][row_frag][col_frag]
    // One uint32 holds two fp16/bf16 elements (one 8×8 mma fragment column).
    uint32_t regs[MMA_STAGES][RF_ROW_FRAGS][RF_COL_FRAGS];

    T      *gmem_ptr;      // this warp's row in global memory
    int64_t gmem_stride;   // elements per gmem row (seq stride)
    T      *smem_base;     // base of the whole tile in smem (for WHOLE_BLOCK reads)
    T      *smem_warp;     // this warp's row chunk in smem  (for per-warp writes)
    int     lane_id;

    // Properties queried by matmul() in gemm.cuh
    static constexpr bool load_entire_block_into_rf = WHOLE_RF;
    static constexpr int  mma_load_stages            = MMA_STAGES;
    static constexpr bool transposed                 = TRANSPOSED;

    // Constructor: split the tile across warps.
    FA_DEVICE TileBuffer(T *gmem_block, int64_t stride, T *smem_tile)
        : lane_id(threadIdx.x % WARP_SIZE),
          gmem_stride(stride),
          smem_base(smem_tile)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        gmem_ptr  = gmem_block + (int64_t)warp_id * GSM_ROWS * stride;
        smem_warp = smem_tile  + warp_id * GSM_ROWS * SMEM_COLS;
    }

    // Zero out all register stages.
    FA_DEVICE_CONSTEXPR void zero() {
        FA_UNROLL for (int s = 0; s < MMA_STAGES; ++s)
        FA_UNROLL for (int r = 0; r < RF_ROW_FRAGS; ++r)
        FA_UNROLL for (int c = 0; c < RF_COL_FRAGS; ++c)
            regs[s][r][c] = 0;
    }

    // Access register array for pipeline stage s.
    FA_DEVICE_CONSTEXPR uint32_t (&data(int s = 0))[RF_ROW_FRAGS][RF_COL_FRAGS] {
        return regs[s];
    }

    // Advance gmem pointer by one full block (for iterating over K/V blocks).
    FA_DEVICE_CONSTEXPR void advance_gmem_block() {
        gmem_ptr += BLOCK_ROWS * gmem_stride;
    }

    // gmem → smem (this warp's row chunk)
    FA_DEVICE_CONSTEXPR void copy_GM2SM() {
        copy_tile_g2s<GSM_ROWS / ROWS_PER_FRAGMENT, SMEM_COLS, SWIZZLED, ASYNC_CP, T>(
            gmem_ptr, smem_warp, gmem_stride, lane_id);
    }

    // smem → gmem (this warp's row chunk)
    FA_DEVICE_CONSTEXPR void copy_SM2GM() {
        copy_tile_s2g<GSM_ROWS / ROWS_PER_FRAGMENT, SMEM_COLS, SWIZZLED, T>(
            gmem_ptr, smem_warp, gmem_stride, lane_id);
    }

    // smem → registers for MMA.
    // stage  : which double-buffer slot to fill (0 or 1).
    // offset : start at this K-fragment column (used in streaming mode).
    FA_DEVICE_CONSTEXPR void copy_SM2RF(int stage = 0, int offset = 0) {
        T *src = WHOLE_BLOCK ? smem_base : smem_warp;
        if constexpr (!TRANSPOSED)
            load_smem2rf<RF_ROW_FRAGS, RF_COL_FRAGS, SMEM_COLS, SWIZZLED, T>(
                regs[stage], src, lane_id, offset);
        else
            load_smem2rf_T<RF_ROW_FRAGS, RF_COL_FRAGS, SMEM_COLS, SWIZZLED, T>(
                regs[stage], src, lane_id, offset);
    }

    // registers → smem (used for O before the final smem→gmem write)
    FA_DEVICE_CONSTEXPR void copy_RF2SM() {
        store_rf2smem<RF_ROW_FRAGS, RF_COL_FRAGS, SMEM_COLS, SWIZZLED, T>(
            regs[0], smem_warp, lane_id);
    }
};

// ---------------------------------------------------------------------------
// AccumTile — float32 accumulator (register-only, no memory operations)
//
// Shape: [ROW_FRAGS][COL_FRAGS * 2]
//   The *2 accounts for mma.m16n8k16 producing 2 floats per 8-col N-fragment.
//   ROW_FRAGS = qo_frags_warp
//   COL_FRAGS = kv_frags  (for S)  or  d_frags  (for O)
// ---------------------------------------------------------------------------
template <int ROW_FRAGS, int COL_FRAGS>
struct AccumTile {
    static constexpr int accum_cols = COL_FRAGS * N_REGS_PER_F32_ACCUM_FRAGMENT;
    float regs[ROW_FRAGS][accum_cols];

    // Dummy constructor (RF only — no memory pointers needed)
    FA_DEVICE AccumTile(void *, int64_t, void *) {}

    FA_DEVICE_CONSTEXPR void zero() {
        FA_UNROLL for (int r = 0; r < ROW_FRAGS; ++r)
        FA_UNROLL for (int c = 0; c < accum_cols; ++c)
            regs[r][c] = 0.f;
    }

    FA_DEVICE_CONSTEXPR float (&data(int = 0))[ROW_FRAGS][accum_cols] {
        return regs;
    }

    // No-ops: accumulator lives entirely in registers
    FA_DEVICE_CONSTEXPR void advance_gmem_block() {}
    FA_DEVICE_CONSTEXPR void copy_GM2SM()          {}
    FA_DEVICE_CONSTEXPR void copy_SM2GM()          {}
    FA_DEVICE_CONSTEXPR void copy_SM2RF(int = 0, int = 0) {}
    FA_DEVICE_CONSTEXPR void copy_RF2SM()          {}

    static constexpr bool load_entire_block_into_rf = false;
    static constexpr int  mma_load_stages            = 1;
    static constexpr bool transposed                 = false;
};

// ---------------------------------------------------------------------------
// RFTile — fp16/bf16 register-only tile (no gmem/smem operations)
//
// Used for P_b16: the fp16 version of the S attention score matrix.
// Written via convert_to_16_bit_dtype(), read via data() inside matmul().
// Shape: [ROW_FRAGS][COL_FRAGS]
// ---------------------------------------------------------------------------
template <int ROW_FRAGS, int COL_FRAGS, typename T>
struct RFTile {
    uint32_t regs[ROW_FRAGS][COL_FRAGS];

    // Dummy constructor (RF only)
    FA_DEVICE RFTile(void *, int64_t, void *) {}

    FA_DEVICE_CONSTEXPR void zero() {
        FA_UNROLL for (int r = 0; r < ROW_FRAGS; ++r)
        FA_UNROLL for (int c = 0; c < COL_FRAGS; ++c)
            regs[r][c] = 0;
    }

    FA_DEVICE_CONSTEXPR uint32_t (&data(int = 0))[ROW_FRAGS][COL_FRAGS] {
        return regs;
    }

    // No-ops: register-only tile
    FA_DEVICE_CONSTEXPR void advance_gmem_block() {}
    FA_DEVICE_CONSTEXPR void copy_GM2SM()          {}
    FA_DEVICE_CONSTEXPR void copy_SM2GM()          {}
    FA_DEVICE_CONSTEXPR void copy_SM2RF(int = 0, int = 0) {}
    FA_DEVICE_CONSTEXPR void copy_RF2SM()          {}

    // P is fully loaded into RF via convert_to_16_bit_dtype() before the loop
    static constexpr bool load_entire_block_into_rf = true;
    static constexpr int  mma_load_stages            = 1;
    static constexpr bool transposed                 = false;
};

// ---------------------------------------------------------------------------
// RFVector — 1-D per-fragment scalar array (used for row statistics m and l)
// ---------------------------------------------------------------------------
template <typename value_t, int N>
struct RFVector {
    static constexpr int size = N;
    value_t regs[N];
    FA_DEVICE_CONSTEXPR value_t &operator[](int idx) { return regs[idx]; }
};

} // namespace flash
