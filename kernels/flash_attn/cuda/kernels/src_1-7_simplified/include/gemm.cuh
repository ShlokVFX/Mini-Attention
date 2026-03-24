#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "common.cuh"
#include "ptx_functions.cuh"

// =============================================================================
// gemm.cuh — warp-level matrix multiply-accumulate (mma.m16n8k16)
//
// Two public interfaces:
//
//   MMAConfig<A, B, C, K_TOTAL, K_STEP, val_t>
//     Compile-time descriptor for one GEMM operation (QK or PV).
//     Holds matrix tile types and the double-buffer flags.
//
//   matmul<MMAConfig>(A, B, C)
//     Executes the warp-level GEMM: C += A * B
//     Handles both "load everything into RF first" and "stream K tiles"
//     patterns, plus optional double-buffering.
// =============================================================================

namespace flash {

// ---------------------------------------------------------------------------
// MMA tile dimensions for mma.sync.m16n8k16
// ---------------------------------------------------------------------------
#define MMA_M   16
#define MMA_N    8
#define MMA_K   16

// Fragments processed per mma instruction (in terms of LDMATRIX_MAT_SIZE tiles)
#define MMA_M_FRAGS_PER_INST  2   // MMA_M / LDMATRIX_MAT_SIZE = 16/8
#define MMA_N_FRAGS_PER_INST  1   // MMA_N / LDMATRIX_MAT_SIZE =  8/8
#define MMA_K_FRAGS_PER_INST  2   // MMA_K / LDMATRIX_MAT_SIZE = 16/8

// ---------------------------------------------------------------------------
// mma_f32_accum: core warp-level outer-product accumulation
//
// Computes a tile of C += A_slice * B_slice for one outer-product step.
// Iterates over the K dimension in steps of MMA_K_FRAGS_PER_INST (= 2).
//
// Template params:
//   val_t      — half or nv_bfloat16 (input element type)
//   M_frags    — A row fragments (= qo_frags_warp for QK, = qo_frags_warp for PV)
//   N_frags    — B row fragments (= kv_frags for QK, = d_frags for PV)
//   K_frags_A  — A col fragments (K tiles loaded in A)
//   K_frags_B  — B col fragments (K tiles loaded in B)
//
// a_col_off / b_col_off: start offset within the full K dimension.
//   Used when one operand is entirely in RF and we index into different K slices.
// ---------------------------------------------------------------------------
template <typename val_t,
          int M_frags, int N_frags,
          int K_frags_A, int K_frags_B>
FA_DEVICE_CONSTEXPR void
mma_f32_accum(uint32_t (&A)[M_frags][K_frags_A],
              uint32_t (&B)[N_frags][K_frags_B],
              float    (&C)[M_frags][N_frags * N_REGS_PER_F32_ACCUM_FRAGMENT],
              int a_col_off = 0, int b_col_off = 0) {

    constexpr int K_iters = constexpr_min(K_frags_A, K_frags_B);

    FA_UNROLL
    for (int k = 0; k < K_iters; k += MMA_K_FRAGS_PER_INST) {
        FA_UNROLL
        for (int m = 0; m < M_frags; m += MMA_M_FRAGS_PER_INST) {
            FA_UNROLL
            for (int n = 0; n < N_frags; n += MMA_N_FRAGS_PER_INST) {
                // C[m:m+2][n*2 : n*2+2] += A[m:m+2][k:k+2] * B[n][k:k+2]
                mma_m16n8k16_f32_accum<val_t>(
                    C[m    ][n * 2],     C[m    ][n * 2 + 1],
                    C[m + 1][n * 2],     C[m + 1][n * 2 + 1],
                    A[m    ][k + a_col_off],
                    A[m + 1][k + a_col_off],
                    A[m    ][k + 1 + a_col_off],
                    A[m + 1][k + 1 + a_col_off],
                    B[n][k + b_col_off],
                    B[n][k + 1 + b_col_off],
                    C[m    ][n * 2],     C[m    ][n * 2 + 1],
                    C[m + 1][n * 2],     C[m + 1][n * 2 + 1]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MMAConfig — compile-time descriptor for one GEMM
//
// Template params:
//   _A, _B, _C — TileBuffer / AccumTile / RFTile types
//   K_TOTAL    — total K fragments to cover (d_frags for QK; kv_frags for PV)
//   K_STEP     — K fragments loaded per matmul step (0 means load all at once)
//   val_t      — fp16 or bf16
// ---------------------------------------------------------------------------
template <typename _A, typename _B, typename _C,
          int K_TOTAL, int K_STEP, typename val_t>
struct MMAConfig {
    using A = _A;
    using B = _B;
    using C = _C;
    using value_t = val_t;

    static constexpr int k_total  = K_TOTAL;
    static constexpr int k_step   = K_STEP;

    // Double-buffer flags: prefetch next K slice while executing current mma
    static constexpr bool db_A = !A::load_entire_block_into_rf && A::mma_load_stages > 1;
    static constexpr bool db_B = !B::load_entire_block_into_rf && B::mma_load_stages > 1;
    static constexpr bool db   = db_A || db_B;
};

// ---------------------------------------------------------------------------
// matmul: execute C += A * B for one full tile
//
// Handles four cases transparently:
//   1. A & B both pre-loaded into RF (simplest — direct mma loop)
//   2. A pre-loaded, B streamed  (common for Q×K when Q is whole-RF)
//   3. A streamed, B pre-loaded  (common for P×V)
//   4. Both streamed with optional double-buffer
// ---------------------------------------------------------------------------
template <typename MMA>
FA_DEVICE_CONSTEXPR void
matmul(typename MMA::A &A, typename MMA::B &B, typename MMA::C &C) {

    using A_t = typename MMA::A;
    using B_t = typename MMA::B;
    using value_t = typename MMA::value_t;

    // Stage toggle mask (0 if single-stage, 1 if double-buffer)
    constexpr int A_toggle = A_t::mma_load_stages - 1;
    constexpr int B_toggle = B_t::mma_load_stages - 1;

    int A_stage = 0, B_stage = 0;

    // Pre-load first K slice for double-buffered operands
    if constexpr (MMA::db_A) A.copy_SM2RF(A_stage);
    if constexpr (MMA::db_B) B.copy_SM2RF(B_stage);

    FA_UNROLL
    for (int k = 0; k < MMA::k_total; k += MMA::k_step) {

        // Prefetch the NEXT K slice into the alternate buffer stage
        if constexpr (!A_t::load_entire_block_into_rf || !B_t::load_entire_block_into_rf) {
            int k_next = k + (MMA::db ? MMA::k_step : 0);
            if (k_next < MMA::k_total) {
                if constexpr (!A_t::load_entire_block_into_rf)
                    A.copy_SM2RF(A_toggle ^ A_stage, k_next);
                if constexpr (!B_t::load_entire_block_into_rf)
                    B.copy_SM2RF(B_toggle ^ B_stage, k_next);
            }
        }

        // Column offsets: when the block is fully in RF, index into the right slice
        int a_col_off = A_t::load_entire_block_into_rf ? k : 0;
        int b_col_off = B_t::load_entire_block_into_rf ? k : 0;

        mma_f32_accum<value_t>(A.data(A_stage), B.data(B_stage), C.data(),
                               a_col_off, b_col_off);

        A_stage ^= A_toggle;
        B_stage ^= B_toggle;
    }
}

} // namespace flash
