#pragma once

// =============================================================================
// common.cuh — shared constants, macros, and tiny helpers
// Merges common.h + utils.h from src_1-7.
// =============================================================================

// ---------------------------------------------------------------------------
// Device / host function qualifiers
// ---------------------------------------------------------------------------
#define FA_UNROLL           _Pragma("unroll")
#define FA_DEVICE           __forceinline__ __device__
#define FA_DEVICE_CONSTEXPR __forceinline__ __device__ constexpr

// ---------------------------------------------------------------------------
// Warp constants
// ---------------------------------------------------------------------------
#define WARP_SIZE               32
#define SHFL_ENTIRE_WARP_MASK   0xffffffff

// ---------------------------------------------------------------------------
// Memory access width: 128-bit / 16-byte vectorized loads (uint4)
// ---------------------------------------------------------------------------
#define B16_BYTES               2                                    // bytes per fp16 element
#define BYTES_PER_VEC4_ACCESS   16                                   // bytes per uint4 load
#define ELEMS_PER_VEC4_ACCESS   (BYTES_PER_VEC4_ACCESS / B16_BYTES)  // = 8 fp16 elements

// ---------------------------------------------------------------------------
// Fragment / tile sizes for ldmatrix + mma.m16n8k16
//   One ldmatrix "fragment" covers an (8×8) sub-tile of the operand.
//   ROWS_PER_FRAGMENT = 8 (rows per ldmatrix tile)
//   COLS_PER_FRAGMENT = 8 (cols per ldmatrix tile)
// ---------------------------------------------------------------------------
#define LDMATRIX_MAT_SIZE       8
#define ROWS_PER_FRAGMENT       LDMATRIX_MAT_SIZE
#define COLS_PER_FRAGMENT       LDMATRIX_MAT_SIZE

// mma.m16n8k16 accumulator: each (8-row) N-fragment produces 2 float32 output regs
#define N_REGS_PER_F32_ACCUM_FRAGMENT  2

// mma operand register layout (informational; used in gemm.cuh)
#define MMA_A_REGS_PER_ROW  2
#define MMA_A_REGS_PER_COL  2
#define MMA_B_REGS_PER_ROW  2
#define MMA_B_REGS_PER_COL  1
#define MMA_C_REGS_PER_ROW  1
#define MMA_C_REGS_PER_COL  2

// ---------------------------------------------------------------------------
// gmem <-> smem tile copy granularity
//   Each copy iteration handles 4 rows of the tile in parallel across a warp.
// ---------------------------------------------------------------------------
#define GSM_LDST_ROWS_PER_ITER  4

// ---------------------------------------------------------------------------
// Double-buffer pipeline stages
// ---------------------------------------------------------------------------
#define N_BUFFER_STAGES  2

// ---------------------------------------------------------------------------
// Compile-time helpers (in namespace flash so they don't pollute global scope)
// ---------------------------------------------------------------------------
namespace flash {

// min of two compile-time integers
constexpr int constexpr_min(int a, int b) { return (a < b) ? a : b; }

// floor(log2(n)) for power-of-2 checks
constexpr int constexpr_log2_floor(int n) { return std::__bit_width(n) - 1; }

} // namespace flash
