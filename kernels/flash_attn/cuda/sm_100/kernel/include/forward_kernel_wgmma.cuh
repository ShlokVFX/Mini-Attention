#pragma once

#include <cuda/std/limits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common.cuh"
#include "flash_attention.cuh"
#include "gemm.cuh"
#include "kernel_config.cuh"
#include "ptx_functions.cuh"
#include "softmax.cuh"

// =============================================================================
// forward_kernel_wgmma.cuh — Flash Attention forward pass using WGMMA
//
// B200 / SM_100 WGMMA optimizations over the baseline mma.sync kernel:
//
//   1. QK GEMM uses wgmma.mma_async instead of ldmatrix + mma.sync.
//      All 128 threads in the warpgroup issue the same wgmma instruction;
//      hardware routes the correct Q/K rows from smem to each warp.
//      This eliminates the ldmatrix bottleneck and improves instruction
//      issue density.
//
//   2. Q stays in shared memory (not pre-loaded into registers).
//      The wgmma descriptor points directly to Q in smem.
//
//   3. PV GEMM still uses mma.sync because P (fp16 S after softmax) lives
//      in registers, not smem, so wgmma cannot read it directly.
//
//   4. Register layout compatibility: wgmma m64nNk16 output is distributed
//      across the 128-thread warpgroup with the same per-warp, per-thread
//      layout as stacked mma.sync m16n8k16 calls. The softmax, rescaling,
//      and output-store code therefore requires no modification.
//
// Constraints:
//   - n_warps must be a multiple of WARPS_PER_WARPGROUP (4).
//   - KC::Bc must equal the wgmma N-tile (64 or 128).
//   - KC::Q_t::WHOLE_RF must be false (Q must stay in smem for wgmma).
//     We use Q_K=0 path and skip SM2RF for Q, relying on the smem descriptor.
//
// Usage: instantiate flash_forward_kernel_wgmma<KC> alongside the existing
//        flash_forward_kernel<KC> in flash_kernels.cuh.
// =============================================================================

namespace flash {

// ---------------------------------------------------------------------------
// wgmma_qk_loop: compute S += Q * K^T using WGMMA for one KV block.
//
// Template parameters:
//   KC        — KernelConfig<...>
//   BC        — Bc tile width (must be 64 or 128; determines wgmma N shape)
//
// Per-warpgroup: handles WGMMA_M=64 rows of Q.
// Loops over d_head in WGMMA_K=16 element steps.
// ---------------------------------------------------------------------------
template <typename KC>
FA_DEVICE void wgmma_qk_loop(
    typename KC::value_t *smem_Q_wg,   // this warpgroup's Q rows in smem
    typename KC::value_t *smem_K,      // K block in smem (whole Bc×D tile)
    float                 S[][KC::kv_accum_regs])  // S accumulator per warp
{
    using value_t = typename KC::value_t;
    static_assert(std::is_same_v<value_t, half>,
                  "WGMMA path currently only supports FP16");

    // ld_bytes: row stride in bytes for Q and K smem tiles (same for both).
    //   Q is (Br_wg × D) row-major  → row stride = D * sizeof(fp16) = 256
    //   K is (Bc   × D) row-major  → row stride = D * sizeof(fp16) = 256
    constexpr int ld_bytes = KC::d_head * (int)sizeof(value_t);  // = 256

    // Flatten the [qo_frags_warp][kv_accum_regs] accumulator to a 1-D array.
    constexpr int FLAT_SIZE = KC::qo_frags_warp * KC::kv_accum_regs;
    static_assert(FLAT_SIZE == 32 || FLAT_SIZE == 64,
                  "WGMMA path requires flat S size of 32 (Bc=64) or 64 (Bc=128)");

    float *flat = &S[0][0];

    // Fence before wgmma: required when registers were previously modified
    // by mma.sync or explicit writes (e.g. zeroing the accumulator).
    wgmma_fence();

    // Each WGMMA step covers a 64×WGMMA_K=16 sub-tile of Q (columns) and
    // a WGMMA_K×Bc sub-tile of K^T (rows of K), iterating over d_head=128.
    FA_UNROLL
    for (int k = 0; k < KC::d_head / WGMMA_K; ++k) {
        // A descriptor: Q[wg_rows, k*16 .. k*16+15]
        //   Byte offset from smem_Q_wg start: k * WGMMA_K elements (column advance)
        uint64_t descA = build_smem_desc(smem_Q_wg + k * WGMMA_K, ld_bytes);

        // B descriptor (with imm_trans_b=1): K[k*16..k*16+15, 0..Bc-1]
        //   Row advance: k * WGMMA_K rows, each row = d_head elements
        //   Byte offset: k * WGMMA_K * d_head elements from smem_K start
        uint64_t descB = build_smem_desc(smem_K + k * WGMMA_K * KC::d_head, ld_bytes);

        const int scale_d = (k > 0) ? 1 : 0;

        if constexpr (FLAT_SIZE == 32) {
            float (&d32)[32] = reinterpret_cast<float (&)[32]>(*flat);
            wgmma_m64n64k16_f16(d32, descA, descB, scale_d);
        } else {
            float (&d64)[64] = reinterpret_cast<float (&)[64]>(*flat);
            wgmma_m64n128k16_f16(d64, descA, descB, scale_d);
        }
    }

    wgmma_commit();
    wgmma_wait<0>();
}


// ---------------------------------------------------------------------------
// flash_forward_kernel_wgmma: main WGMMA kernel
//
// Uses wgmma for QK GEMM and mma.sync for PV GEMM (P stays in registers).
// ---------------------------------------------------------------------------
template <typename KC>
__global__ void
flash_forward_kernel_wgmma(__grid_constant__ const ForwardKernelArgs args) {

    using value_t = typename KC::value_t;
    using accum_t = typename KC::accum_t;
    using index_t = int64_t;

    // ---- CTA identity ----
    const int sample      = blockIdx.z;
    const int head        = blockIdx.y;
    const int q_seq_block = blockIdx.x;

    // ---- Warpgroup identity ----
    // n_warps must be a multiple of WARPS_PER_WARPGROUP (= 4).
    static_assert(KC::n_warps % WARPS_PER_WARPGROUP == 0,
                  "WGMMA kernel requires n_warps to be a multiple of 4");
    constexpr int n_warpgroups = KC::n_warps / WARPS_PER_WARPGROUP;

    const int wg_id      = threadIdx.x / WARPGROUP_SIZE;   // warpgroup index [0..n_warpgroups)
    const int warp_id    = (threadIdx.x / WARP_SIZE) % WARPS_PER_WARPGROUP;
    const int lane_id    = threadIdx.x % WARP_SIZE;

    // Rows of Q handled by this warpgroup
    constexpr int wg_br  = KC::Br / n_warpgroups;           // = Br / n_warpgroups

    const index_t gmem_seq_stride = args.seq_stride;
    const index_t sample_head_off =
        sample * args.batch_stride + head * args.head_stride;

    const index_t QO_off =
        sample_head_off + q_seq_block * KC::Br * gmem_seq_stride;
    const index_t KV_off = sample_head_off;

    value_t *gmem_Q = &static_cast<value_t *>(args.Q)[QO_off];
    value_t *gmem_O = &static_cast<value_t *>(args.O)[QO_off];
    value_t *gmem_K = &static_cast<value_t *>(args.K)[KV_off];
    value_t *gmem_V = &static_cast<value_t *>(args.V)[KV_off];

    // ---- Shared memory layout ----
    extern __shared__ __align__(16) char ch_smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(ch_smem);
    value_t *smem_K = &smem_Q[KC::Br * KC::d_head];
    value_t *smem_V = &smem_K[KC::Bc * KC::d_head];

    // This warpgroup's slice of smem_Q (wg_br rows starting at wg_id*wg_br).
    value_t *smem_Q_wg = smem_Q + wg_id * wg_br * KC::d_head;

    // ---- Tile objects for loading ----
    // Q and V use the standard TileBuffer infrastructure.
    // K also uses TileBuffer but we won't load it into RF — wgmma reads from smem.
    typename KC::Q_t     Q(gmem_Q, gmem_seq_stride, smem_Q);
    typename KC::K_t     K(gmem_K, gmem_seq_stride, smem_K);
    typename KC::V_t     V(gmem_V, gmem_seq_stride, smem_V);

    typename KC::S_acc_t S_acc(nullptr, -1, nullptr);
    typename KC::P_t     P_b16(nullptr, -1, nullptr);
    typename KC::O_acc_t O_acc(nullptr, -1, nullptr);
    typename KC::O_val_t O_b16(gmem_O, gmem_seq_stride, smem_Q);  // Q/O share smem

    // ---- Issue async copies: Q first ----
    Q.copy_GM2SM();
    cp_async_commit<KC::async_copy>();

    if constexpr (KC::eager_load) {
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit<KC::async_copy>();
    }

    O_acc.zero();

    // ---- Softmax statistics ----
    const accum_t softmax_scale =
        rsqrtf(static_cast<accum_t>(KC::d_head)) *
        (KC::opt_softmax ? M_LOG2E : 1.0f);

    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();

    accum_t m[KC::qo_frags_warp];
    accum_t l[KC::qo_frags_warp];
    FA_UNROLL
    for (int q = 0; q < KC::qo_frags_warp; ++q) {
        m[q] = neg_inf;
        l[q] = 0.0f;
    }

    // Wait for Q to arrive in smem (WGMMA reads Q from smem — no SM2RF load)
    if constexpr (KC::eager_load) {
        cp_async_wait<1, KC::async_copy>();
    } else {
        cp_async_wait<0, KC::async_copy>();
    }
    __syncthreads();

    // ---- Main loop: iterate over KV blocks ----
    for (int j = 0; j < args.n_KV_blocks; ++j) {

        if constexpr (!KC::eager_load) {
            K.copy_GM2SM();
            K.advance_gmem_block();
            cp_async_commit<KC::async_copy>();
        }

        S_acc.zero();

        // Wait for K tile
        cp_async_wait<0, KC::async_copy>();
        __syncthreads();

        if constexpr (KC::eager_load) {
            // Start loading V while computing QK
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<KC::async_copy>();
        }

        // ---- QK GEMM via WGMMA (B200-specific) ----
        // Each warpgroup computes its 64-row slice of S = Q * K^T
        wgmma_qk_loop<KC>(smem_Q_wg, smem_K, S_acc.data());

        // Wait for V tile
        cp_async_wait<0, KC::async_copy>();
        __syncthreads();

        if constexpr (KC::eager_load) {
            if (j < args.n_KV_blocks - 1) {
                K.copy_GM2SM();
                K.advance_gmem_block();
                cp_async_commit<KC::async_copy>();
            }
        }

        // ---- Online softmax (unchanged from mma.sync path) ----
        accum_t m_next[KC::qo_frags_warp];

        if constexpr (!KC::opt_softmax) {
            scale_S_accum(S_acc.data(), softmax_scale);
        }
        calc_row_max(S_acc.data(), m_next, m);
        scale_l_O<KC::opt_softmax>(m_next, m, l, O_acc.data(), softmax_scale);
        exponentiate_tensor<KC::opt_softmax>(S_acc.data(), m_next, softmax_scale);
        update_row_exp_sum(S_acc.data(), l);

        // Convert fp32 S → fp16/bf16 P for PV matmul
        convert_to_16_bit_dtype<value_t>(S_acc.data(), P_b16.data());

        if constexpr (!KC::eager_load) {
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<KC::async_copy>();
            cp_async_wait<0, KC::async_copy>();
            __syncthreads();
        }

        if constexpr (KC::V_t::load_entire_block_into_rf) {
            V.copy_SM2RF();
        }

        // ---- PV GEMM via mma.sync (P is in registers) ----
        matmul<typename KC::PV_GEMM>(P_b16, V, O_acc);
    }

    // ---- Final softmax normalization and output write-back ----
    final_softmax_normalization(O_acc.data(), l);

    convert_to_16_bit_dtype<value_t>(O_acc.data(), O_b16.data());

    O_b16.copy_RF2SM();
    __syncwarp();

    O_b16.copy_SM2GM();
}

} // namespace flash
