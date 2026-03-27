#pragma once

// =============================================================================
// common.cuh — shared constants, macros, and tiny helpers
// B200 (SM_100 / sm_100) variant.
// =============================================================================

// ---------------------------------------------------------------------------
// Device / host function qualifiers
// ---------------------------------------------------------------------------
#define FA_UNROLL           _Pragma("unroll")
#define FA_DEVICE           __forceinline__ __device__
#define FA_DEVICE_CONSTEXPR __forceinline__ __device__ constexpr

// ---------------------------------------------------------------------------
// Warp / warpgroup constants
// ---------------------------------------------------------------------------
#define WARP_SIZE               32
#define WARPGROUP_SIZE          128      // 4 warps per warpgroup (WGMMA)
#define SHFL_ENTIRE_WARP_MASK   0xffffffff

// ---------------------------------------------------------------------------
// Memory access width: 128-bit / 16-byte vectorized loads (uint4)
// ---------------------------------------------------------------------------
#define B16_BYTES               2
#define BYTES_PER_VEC4_ACCESS   16
#define ELEMS_PER_VEC4_ACCESS   (BYTES_PER_VEC4_ACCESS / B16_BYTES)  // = 8 fp16

// ---------------------------------------------------------------------------
// Fragment / tile sizes for ldmatrix + mma.m16n8k16
// ---------------------------------------------------------------------------
#define LDMATRIX_MAT_SIZE       8
#define ROWS_PER_FRAGMENT       LDMATRIX_MAT_SIZE
#define COLS_PER_FRAGMENT       LDMATRIX_MAT_SIZE

// mma.m16n8k16 accumulator: 2 float32 registers per 8-column N-fragment
#define N_REGS_PER_F32_ACCUM_FRAGMENT  2

// mma operand register layout
#define MMA_A_REGS_PER_ROW  2
#define MMA_A_REGS_PER_COL  2
#define MMA_B_REGS_PER_ROW  2
#define MMA_B_REGS_PER_COL  1
#define MMA_C_REGS_PER_ROW  1
#define MMA_C_REGS_PER_COL  2

// ---------------------------------------------------------------------------
// gmem <-> smem tile copy granularity
// ---------------------------------------------------------------------------
#define GSM_LDST_ROWS_PER_ITER  4

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------
#define N_BUFFER_STAGES  2

// ---------------------------------------------------------------------------
// B200 (SM_100) hardware limits
//   - Maximum addressable shared memory per CTA: 228 KB
//   - L2 cache sector width: 256 bytes (same as SM_120 consumer Blackwell)
//   - WGMMA: warpgroup (128 threads) matrix multiply, available since SM_90
// ---------------------------------------------------------------------------
#define B200_MAX_SMEM_BYTES     (228 * 1024)   // 228 KB per SM on B200

// ---------------------------------------------------------------------------
// WGMMA tile dimensions (warpgroup-level matrix multiply)
//   wgmma.mma_async.sync.aligned.m64nNk16  — one warpgroup (128 threads)
//   computes a 64×N tile per step across the K=16 dimension.
// ---------------------------------------------------------------------------
#define WGMMA_M              64   // fixed M for wgmma.m64nNk16
#define WGMMA_K              16   // fixed K step per wgmma instruction
#define WARPS_PER_WARPGROUP   4   // 4 warps × 32 threads = 128 threads

// ---------------------------------------------------------------------------
// Compile-time helpers
// ---------------------------------------------------------------------------
namespace flash {

constexpr int constexpr_min(int a, int b) { return (a < b) ? a : b; }
constexpr int constexpr_log2_floor(int n) { return std::__bit_width(n) - 1; }

} // namespace flash
