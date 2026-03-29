#pragma once

#include "common.cuh"
#include "flash_attention.cuh"
#include "gemm.cuh"
#include "load_store.cuh"
#include "tensor.cuh"

// =============================================================================
// kernel_config.cuh — compile-time kernel configuration
//
// KEY SIMPLIFICATION vs src_1-7/include/static_kernel_configuration.cuh:
//
//   Original code used a two-layer structure:
//     ForwardKernelTileShapes<FlashForwardKernelConfig CFG>  (nested N:: namespace)
//     StaticForwardKernelConfig<FlashForwardKernelConfig CFG>
//
//   Both took a struct (FlashForwardKernelConfig) as a non-type template
//   parameter — a C++20 feature borrowed from CUTLASS.
//
//   Here we replace both structs with a SINGLE flat KernelConfig<...> that:
//     • takes 13 plain int/bool template parameters (no struct NTTPs)
//     • exposes every tile dimension as a named constexpr int directly
//     • defines the matrix tile types (Q, K, V, P, O_acc, S_acc, O_val)
//       and GEMM descriptors (QK_GEMM, PV_GEMM) in one place
//     • is directly usable as the Kernel template parameter to flash_forward_kernel
//
//   To instantiate a kernel you just write:
//     KernelConfig<torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false>
//   instead of:
//     StaticForwardKernelConfig<FlashForwardKernelConfig{torch::kFloat16, ...}>
// =============================================================================

namespace flash {

// ---------------------------------------------------------------------------
// Validate MMA K-load fragment count at compile time.
//   n must be a power-of-2 > 1, and must fit within the available fragments.
// ---------------------------------------------------------------------------
template <int n, int K, bool double_buffer>
constexpr void check_mma_load_k() {
    static_assert(((n & (n - 1)) == 0) && n != 1,
                  "MMA K-load fragment count must be power-of-2 and != 1");
    constexpr int max_frags = (double_buffer ? K / 2 : K) / 8;
    static_assert(n <= max_frags, "MMA K-load fragment count exceeds maximum");
}

// ---------------------------------------------------------------------------
// KernelConfig — flat compile-time kernel descriptor
//
// Template parameters match FlashForwardKernelConfig field order:
//   DTYPE     — torch::kFloat16 or torch::kBFloat16
//   D         — head dimension (64 or 128)
//   BR        — query tile rows (64 or 128)
//   BC        — key/value tile rows (32, 64, or 128)
//   NW        — warps per CTA (4 or 8)
//   ASYNC     — use cp.async for gmem→smem
//   EAGER     — pre-load next K/V block as early as possible
//   SWZ       — XOR swizzle smem addresses
//   Q_K       — Q MMA inter-step K fragments  (0 = load all at once)
//   K_K       — K MMA inter-step K fragments  (0 = load all at once)
//   V_K       — V MMA inter-step K fragments  (0 = load all at once)
//   DBUF      — double-buffer MMA loads
//   OSOFTMAX  — optimized softmax (exp2 path)
// ---------------------------------------------------------------------------
template <
    at::ScalarType DTYPE,
    int D,  int BR, int BC, int NW,
    bool ASYNC, bool EAGER, bool SWZ,
    int  Q_K,   int  K_K,   int  V_K,
    bool DBUF,  bool OSOFTMAX
>
struct KernelConfig {
    // -------------------------------------------------------------------------
    // Element types
    // -------------------------------------------------------------------------
    using value_t = std::conditional_t<DTYPE == at::kBFloat16, nv_bfloat16, half>;
    using accum_t = float;

    // -------------------------------------------------------------------------
    // Core dimensions (directly accessible, no N:: indirection)
    // -------------------------------------------------------------------------
    static constexpr int d_head  = D;
    static constexpr int Br      = BR;
    static constexpr int Bc      = BC;
    static constexpr int n_warps = NW;

    // -------------------------------------------------------------------------
    // Fragment counts
    //   One ldmatrix/mma fragment covers ROWS_PER_FRAGMENT (8) rows or cols.
    // -------------------------------------------------------------------------

    // Number of (8-col) fragments across the head dimension
    static constexpr int d_frags        = D / ROWS_PER_FRAGMENT;

    // Q/O: each warp handles (Br / n_warps) rows
    static constexpr int qo_rows_warp   = BR / NW;
    static constexpr int qo_frags_warp  = qo_rows_warp / ROWS_PER_FRAGMENT;

    // K/V: full Bc block for computation; split across warps for load/store
    static constexpr int kv_frags       = BC / ROWS_PER_FRAGMENT;   // full block
    static constexpr int kv_frags_warp  = kv_frags / NW;            // per-warp load
    static constexpr int kv_rows_warp   = kv_frags_warp * ROWS_PER_FRAGMENT;

    // f32 accumulator register count per fragment row
    // (2 floats per mma N-fragment — mma.m16n8k16 output layout)
    static constexpr int d_accum_regs   = d_frags  * N_REGS_PER_F32_ACCUM_FRAGMENT;
    static constexpr int kv_accum_regs  = kv_frags * N_REGS_PER_F32_ACCUM_FRAGMENT;

    // -------------------------------------------------------------------------
    // MMA K-loading: how many K-fragments to load between mma instructions.
    //   0 means "load the entire block into RF before the mma loop starts".
    //   Non-zero means "stream this many fragments per step".
    // -------------------------------------------------------------------------
    static constexpr int q_k_frags = (Q_K == 0) ? d_frags  : Q_K;
    static constexpr int k_k_frags = (K_K == 0) ? d_frags  : K_K;
    static constexpr int v_k_frags = (V_K == 0) ? kv_frags : V_K;

    // Pipeline stages: 2 if streaming + double-buffer enabled, else 1
    static constexpr int q_stages = (Q_K > 0 && DBUF) ? 2 : 1;
    static constexpr int k_stages = (K_K > 0 && DBUF) ? 2 : 1;
    static constexpr int v_stages = (V_K > 0 && DBUF) ? 2 : 1;

    // -------------------------------------------------------------------------
    // Feature flags
    // -------------------------------------------------------------------------
    static constexpr bool async_copy    = ASYNC;
    static constexpr bool eager_load    = EAGER;
    static constexpr bool swizzled      = SWZ;
    static constexpr bool double_buf    = DBUF;
    static constexpr bool opt_softmax   = OSOFTMAX;

    // Shared memory: Q tile + K tile + V tile
    static constexpr int smem_bytes = (BR + BC * 2) * D * (int)sizeof(value_t);

    // -------------------------------------------------------------------------
    // Matrix tile types
    //   These replace the using Q_t / K_t / V_t aliases that were deeply
    //   nested inside StaticForwardKernelConfig in src_1-7.
    // -------------------------------------------------------------------------

    // Q: (Br, D) tile — each warp loads its own row chunk; pre-load all into RF
    //    when Q_K == 0 (whole-RF mode), otherwise stream K slices.
    using Q_t = TileBuffer<
        qo_rows_warp,    // GSM_ROWS
        qo_frags_warp,   // RF_ROW_FRAGS
        q_k_frags,       // RF_COL_FRAGS  (= K slices per mma step)
        D,               // SMEM_COLS
        BR,              // BLOCK_ROWS
        false,           // WHOLE_BLOCK = false (Q uses per-warp smem)
        (Q_K == 0),      // WHOLE_RF: load entire Q block into RF at start
        q_stages,        // MMA pipeline stages
        false,           // TRANSPOSED = false
        ASYNC, SWZ, value_t>;

    // K: (Bc, D) tile — entire block used for computation (WHOLE_BLOCK=true)
    using K_t = TileBuffer<
        kv_rows_warp,
        kv_frags,        // RF_ROW_FRAGS = full Bc block (all rows for compute)
        k_k_frags,
        D,
        BC,
        true,            // WHOLE_BLOCK = true (compute over entire K block)
        (K_K == 0),
        k_stages,
        false,
        ASYNC, SWZ, value_t>;

    // V: (Bc, D) tile — transposed ldmatrix load; d_head becomes the row dim
    using V_t = TileBuffer<
        kv_rows_warp,
        d_frags,         // RF_ROW_FRAGS = d_head/8 (output row after transpose)
        v_k_frags,
        D,
        BC,
        true,
        (V_K == 0),
        v_stages,
        true,            // TRANSPOSED = true
        ASYNC, SWZ, value_t>;

    // P: fp16/bf16 version of the S attention score matrix.
    //    Register-only (no gmem/smem ops); written by convert_to_16_bit_dtype.
    using P_t = RFTile<qo_frags_warp, kv_frags, value_t>;

    // O_val: fp16/bf16 output tile (written to smem, then flushed to gmem)
    using O_val_t = TileBuffer<
        qo_rows_warp,
        qo_frags_warp,
        d_frags,         // RF_COL_FRAGS = all d_head frags (whole-RF for O)
        D,
        BR,
        false,
        true,            // WHOLE_RF = true
        1,               // single stage
        false,
        ASYNC, SWZ, value_t>;

    // S_acc: float32 QK attention score accumulator [qo_frags_warp][kv_accum_regs]
    using S_acc_t = AccumTile<qo_frags_warp, kv_frags>;

    // O_acc: float32 PV output accumulator [qo_frags_warp][d_accum_regs]
    using O_acc_t = AccumTile<qo_frags_warp, d_frags>;

    // -------------------------------------------------------------------------
    // GEMM descriptors
    // -------------------------------------------------------------------------

    // QK matmul: S = Q * K^T
    //   K_TOTAL = d_frags (iterate over the head dimension)
    //   K_STEP  = min(q_k_frags, k_k_frags)
    using QK_GEMM = MMAConfig<Q_t, K_t, S_acc_t,
                              d_frags,
                              constexpr_min(q_k_frags, k_k_frags),
                              value_t>;

    // PV matmul: O = P * V
    //   K_TOTAL = kv_frags (iterate over the Bc dimension)
    //   K_STEP  = v_k_frags
    using PV_GEMM = MMAConfig<P_t, V_t, O_acc_t,
                              kv_frags, v_k_frags,
                              value_t>;

    // -------------------------------------------------------------------------
    // Row statistics (softmax m and l vectors, one slot per Q row-fragment)
    // -------------------------------------------------------------------------
    using row_stats_t = RFVector<accum_t, qo_frags_warp>;
};

} // namespace flash
