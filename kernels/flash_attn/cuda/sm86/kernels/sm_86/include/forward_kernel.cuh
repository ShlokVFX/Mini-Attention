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
// forward_kernel.cuh — Flash Attention forward pass kernel
//
// KEY SIMPLIFICATION vs src_1-7:
//   The kernel template parameter is now a KernelConfig<...> (defined in
//   kernel_config.cuh) whose members are accessed directly as KC::d_frags,
//   KC::qo_frags_warp, etc.
//   The nested N:: (ForwardKernelTileShapes) indirection is gone.
// =============================================================================

namespace flash {

template <typename KC>  // KC = KernelConfig<...>
__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {

    using value_t = typename KC::value_t;
    using accum_t = typename KC::accum_t;
    using index_t = int64_t;

    // CTA / block identity
    const int sample       = blockIdx.z;
    const int head         = blockIdx.y;
    const int q_seq_block  = blockIdx.x;

    const index_t gmem_seq_stride = args.seq_stride;

    // Base offset for this (sample, head)
    const index_t sample_head_off =
        sample * args.batch_stride + head * args.head_stride;

    // Q and O share the same gmem block (one B_r-row tile)
    const index_t QO_off =
        sample_head_off + q_seq_block * KC::Br * gmem_seq_stride;
    // K and V start at the beginning of the key sequence
    const index_t KV_off = sample_head_off;

    value_t *gmem_Q = &static_cast<value_t *>(args.Q)[QO_off];
    value_t *gmem_O = &static_cast<value_t *>(args.O)[QO_off];
    value_t *gmem_K = &static_cast<value_t *>(args.K)[KV_off];
    value_t *gmem_V = &static_cast<value_t *>(args.V)[KV_off];

    // -------------------------------------------------------------------------
    // Shared memory layout:  [ Q tile | K tile | V tile ]
    //   Q/O tile : (B_r, D)
    //   K tile   : (B_c, D)
    //   V tile   : (B_c, D)
    // -------------------------------------------------------------------------
    extern __shared__ __align__(16) char ch_smem[];
    value_t *smem_Q = reinterpret_cast<value_t *>(ch_smem);
    value_t *smem_O = smem_Q;                               // Q and O share smem
    value_t *smem_K = &smem_Q[KC::Br * KC::d_head];
    value_t *smem_V = &smem_K[KC::Bc * KC::d_head];

    // -------------------------------------------------------------------------
    // Tile objects — each bundles gmem/smem pointers, register storage, and
    // the copy methods defined in TileBuffer.
    // -------------------------------------------------------------------------
    typename KC::Q_t     Q(gmem_Q, gmem_seq_stride, smem_Q);
    typename KC::K_t     K(gmem_K, gmem_seq_stride, smem_K);
    typename KC::V_t     V(gmem_V, gmem_seq_stride, smem_V);

    // S and P live entirely in registers (no gmem/smem pointers needed)
    typename KC::S_acc_t S_acc(nullptr, -1, nullptr);
    typename KC::P_t     P_b16(nullptr, -1, nullptr);

    // O accumulator: float32, register-only during the loop
    typename KC::O_acc_t O_acc(nullptr, -1, nullptr);
    // O value tile: fp16/bf16, written back to smem then gmem at the end
    typename KC::O_val_t O_b16(gmem_O, gmem_seq_stride, smem_O);

    // -------------------------------------------------------------------------
    // Issue the first async copies
    // -------------------------------------------------------------------------
    Q.copy_GM2SM();
    cp_async_commit<KC::async_copy>();

    if constexpr (KC::eager_load) {
        K.copy_GM2SM();
        K.advance_gmem_block();
        cp_async_commit<KC::async_copy>();
    }

    O_acc.zero();

    // -------------------------------------------------------------------------
    // Initialize softmax statistics
    // -------------------------------------------------------------------------
    const accum_t softmax_scale =
        rsqrtf(static_cast<accum_t>(KC::d_head)) *
        (KC::opt_softmax ? M_LOG2E : 1.0f);

    constexpr accum_t neg_inf = -cuda::std::numeric_limits<float>::infinity();

    // m[q]: running row maximum; l[q]: running row sum of exp
    accum_t m[KC::qo_frags_warp];
    accum_t l[KC::qo_frags_warp];
    FA_UNROLL
    for (int q = 0; q < KC::qo_frags_warp; ++q) {
        m[q] = neg_inf;
        l[q] = 0.0f;
    }

    // -------------------------------------------------------------------------
    // Load Q into registers if we're in whole-RF mode
    // -------------------------------------------------------------------------
    if constexpr (KC::Q_t::load_entire_block_into_rf) {
        if constexpr (KC::eager_load) {
            // Wait only for Q; K is still in flight
            cp_async_wait<1, KC::async_copy>();
        } else {
            cp_async_wait<0, KC::async_copy>();
        }
        // cp_async_wait only blocks the issuing thread; sync the warp
        // since all threads will read Q from smem.
        __syncwarp();
        Q.copy_SM2RF();
    }

    // -------------------------------------------------------------------------
    // Main loop: iterate over K/V blocks
    // -------------------------------------------------------------------------
    for (int j = 0; j < args.n_KV_blocks; ++j) {

        if constexpr (!KC::eager_load) {
            K.copy_GM2SM();
            K.advance_gmem_block();
            cp_async_commit<KC::async_copy>();
        }

        S_acc.zero();

        // Wait for K tile and sync the CTA
        cp_async_wait<0, KC::async_copy>();
        __syncthreads();

        if constexpr (KC::eager_load) {
            // Start loading V while we compute QK
            V.copy_GM2SM();
            V.advance_gmem_block();
            cp_async_commit<KC::async_copy>();
        }

        if constexpr (KC::K_t::load_entire_block_into_rf) {
            K.copy_SM2RF();
        }

        // S = Q * K^T
        matmul<typename KC::QK_GEMM>(Q, K, S_acc);

        // Wait for V tile (we need smem for V now) and sync the CTA
        cp_async_wait<0, KC::async_copy>();
        __syncthreads();

        if constexpr (KC::eager_load) {
            // Pre-load the next K tile (skip on last iteration)
            if (j < args.n_KV_blocks - 1) {
                K.copy_GM2SM();
                K.advance_gmem_block();
                cp_async_commit<KC::async_copy>();
            }
        }

        // ---- Online softmax ----
        accum_t m_next[KC::qo_frags_warp];

        if constexpr (!KC::opt_softmax) {
            scale_S_accum(S_acc.data(), softmax_scale);
        }
        calc_row_max(S_acc.data(), m_next, m);
        scale_l_O<KC::opt_softmax>(m_next, m, l, O_acc.data(), softmax_scale);
        exponentiate_tensor<KC::opt_softmax>(S_acc.data(), m_next, softmax_scale);
        update_row_exp_sum(S_acc.data(), l);

        // Convert float32 S -> fp16/bf16 P for the PV matmul
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

        // O += P * V
        matmul<typename KC::PV_GEMM>(P_b16, V, O_acc);
    }

    // -------------------------------------------------------------------------
    // Final softmax normalization and output write-back
    // -------------------------------------------------------------------------
    final_softmax_normalization(O_acc.data(), l);

    // Convert float32 O_acc -> fp16/bf16
    convert_to_16_bit_dtype<value_t>(O_acc.data(), O_b16.data());

    // Write fp16/bf16 to smem (vectorized stores, fully coalesced)
    O_b16.copy_RF2SM();
    __syncwarp();

    // Flush smem to gmem
    O_b16.copy_SM2GM();
}

} // namespace flash
