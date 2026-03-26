#pragma once
// =============================================================================
// wgmma_forward.cuh — K8 FlashAttention forward kernel using wgmma.mma_async
//
// Fixed config: Br=64, Bc=64, d_head=128, NW=4, fp16.
// One CTA = one warpgroup (4 warps / 128 threads).
//
// vs K7 (mma.sync):
//   - K and V stay in smem; B accessed via matrix descriptor (no ldmatrix→RF for K/V)
//   - QK: wgmma.m64n64k16.f32.f16.f16  transB=1 → reads K^T from K smem
//   - PV: wgmma.m64n128k16.f32.f16.f16 transB=0 → reads V rows from V smem
//   - Q and P in RF as A operand (same ldmatrix layout as mma.sync — wgmma-compatible)
//   - Softmax and O store unchanged (same per-thread accumulator layout)
// =============================================================================

#include <cuda/std/limits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Reuse parent kernel utilities unchanged
#include "../../kernel/include/common.cuh"
#include "../../kernel/include/flash_attention.cuh"
#include "../../kernel/include/ptx_functions.cuh"
#include "../../kernel/include/softmax.cuh"
#include "../../kernel/include/load_store.cuh"
#include "../../kernel/include/swizzling.cuh"
#include "wgmma_ptx.cuh"

namespace flash {

// ---------------------------------------------------------------------------
// K8 compile-time dimensions  (fixed, fp16)
// ---------------------------------------------------------------------------
static constexpr int K8_Br     = 64;
static constexpr int K8_Bc     = 64;
static constexpr int K8_D      = 128;
static constexpr int K8_NW     = 4;

static constexpr int K8_d_frags       = K8_D  / 8;            // 16
static constexpr int K8_qo_rows_warp  = K8_Br / K8_NW;        // 16
static constexpr int K8_qo_frags_warp = K8_qo_rows_warp / 8;  // 2
static constexpr int K8_kv_frags      = K8_Bc / 8;            // 8
static constexpr int K8_kv_frags_warp = K8_kv_frags / K8_NW;  // 2
static constexpr int K8_kv_rows_warp  = K8_kv_frags_warp * 8; // 16

// Smem: Q + K + V  (< 100 KB Blackwell limit)
static constexpr int K8_smem_bytes =
    (K8_Br + K8_Bc * 2) * K8_D * 2;  // 49 152 B

// ---------------------------------------------------------------------------
// wgmma QK: S += Q × K^T  over d_head in 8 k16 steps (transB=1)
//
// Q_regs[2][16]: loaded by load_smem2rf into this warp's 16-row × d_head tile.
// k_desc:        descriptor for K smem tile (Bc×d_head row-major).
//
// A register order per k16 step s:
//   a0 = Q[rf=0][2s],  a1 = Q[rf=1][2s],
//   a2 = Q[rf=0][2s+1],a3 = Q[rf=1][2s+1]
// (matches mma.sync A order: A[row0][k], A[row1][k], A[row0][k+1], A[row1][k+1])
// ---------------------------------------------------------------------------
__forceinline__ __device__ void
k8_wgmma_qk(float (&S)[K8_qo_frags_warp][K8_kv_frags * N_REGS_PER_F32_ACCUM_FRAGMENT],
            uint32_t (&Q)[K8_qo_frags_warp][K8_d_frags],
            uint64_t k_desc)
{
    using namespace k8;
    float (&Sf)[K8_qo_frags_warp * K8_kv_frags * N_REGS_PER_F32_ACCUM_FRAGMENT] =
        reinterpret_cast<float (&)[K8_qo_frags_warp * K8_kv_frags * N_REGS_PER_F32_ACCUM_FRAGMENT]>(S);

    wgmma_fence();
    // Step 0: ScaleD=0 (overwrite)
    wgmma_qk<0>(Sf, Q[0][0], Q[1][0], Q[0][1], Q[1][1], k_desc);
    k_desc = k_desc_advance(k_desc);

    // Steps 1-7: ScaleD=1 (accumulate)
    FA_UNROLL
    for (int s = 1; s < K8_d_frags / 2; ++s) {
        wgmma_qk<1>(Sf, Q[0][2*s], Q[1][2*s], Q[0][2*s+1], Q[1][2*s+1], k_desc);
        k_desc = k_desc_advance(k_desc);
    }
    wgmma_commit();
    wgmma_wait<0>();
}

// ---------------------------------------------------------------------------
// wgmma PV: O += P × V  over Bc in 4 k16 steps (transB=0)
//
// P_regs[2][8]: fp16 attention scores (from convert_to_16_bit_dtype on S).
// v_desc:       descriptor for V smem tile (Bc×d_head row-major).
// ---------------------------------------------------------------------------
__forceinline__ __device__ void
k8_wgmma_pv(float (&O)[K8_qo_frags_warp][K8_d_frags * N_REGS_PER_F32_ACCUM_FRAGMENT],
            uint32_t (&P)[K8_qo_frags_warp][K8_kv_frags],
            uint64_t v_desc)
{
    using namespace k8;
    float (&Of)[K8_qo_frags_warp * K8_d_frags * N_REGS_PER_F32_ACCUM_FRAGMENT] =
        reinterpret_cast<float (&)[K8_qo_frags_warp * K8_d_frags * N_REGS_PER_F32_ACCUM_FRAGMENT]>(O);

    wgmma_fence();
    // Step 0: ScaleD=0
    wgmma_pv<0>(Of, P[0][0], P[1][0], P[0][1], P[1][1], v_desc);
    v_desc = v_desc_advance(v_desc);

    // Steps 1-3: ScaleD=1
    FA_UNROLL
    for (int s = 1; s < K8_kv_frags / 2; ++s) {
        wgmma_pv<1>(Of, P[0][2*s], P[1][2*s], P[0][2*s+1], P[1][2*s+1], v_desc);
        v_desc = v_desc_advance(v_desc);
    }
    wgmma_commit();
    wgmma_wait<0>();
}

// ---------------------------------------------------------------------------
// K8 forward kernel
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(K8_NW * WARP_SIZE, 1)
flash_forward_k8(__grid_constant__ const ForwardKernelArgs args)
{
    using value_t = half;

    const int sample      = blockIdx.z;
    const int head        = blockIdx.y;
    const int q_seq_block = blockIdx.x;
    const int64_t stride  = args.seq_stride;

    const int64_t sh_off = (int64_t)sample * args.batch_stride
                         + (int64_t)head   * args.head_stride;

    value_t *gmem_Q = (value_t*)args.Q + sh_off + q_seq_block * K8_Br * stride;
    value_t *gmem_O = (value_t*)args.O + sh_off + q_seq_block * K8_Br * stride;
    value_t *gmem_K = (value_t*)args.K + sh_off;
    value_t *gmem_V = (value_t*)args.V + sh_off;

    // Smem: [ Q/O (Br×D) | K (Bc×D) | V (Bc×D) ]
    extern __shared__ __align__(16) char ch_smem[];
    value_t *smem_Q = reinterpret_cast<value_t*>(ch_smem);
    value_t *smem_K = smem_Q + K8_Br * K8_D;
    value_t *smem_V = smem_K + K8_Bc * K8_D;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;

    // -----------------------------------------------------------------------
    // Load Q into smem (each warp loads its qo_rows_warp=16 rows)
    // -----------------------------------------------------------------------
    copy_tile_g2s<K8_qo_frags_warp, K8_D, false, true, value_t>(
        gmem_Q + warp_id * K8_qo_rows_warp * stride,
        smem_Q + warp_id * K8_qo_rows_warp * K8_D,
        stride, lane);
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // -----------------------------------------------------------------------
    // Q smem → RF  (ldmatrix_x4, no swizzle)
    // -----------------------------------------------------------------------
    uint32_t Q_regs[K8_qo_frags_warp][K8_d_frags];
    load_smem2rf<K8_qo_frags_warp, K8_d_frags, K8_D, false, value_t>(
        Q_regs,
        smem_Q + warp_id * K8_qo_rows_warp * K8_D,
        lane);

    // -----------------------------------------------------------------------
    // Softmax state
    // -----------------------------------------------------------------------
    const float scale = rsqrtf((float)K8_D) * M_LOG2E;   // exp2 path
    constexpr float neg_inf = -cuda::std::numeric_limits<float>::infinity();

    float m[K8_qo_frags_warp], l[K8_qo_frags_warp];
    FA_UNROLL for (int q = 0; q < K8_qo_frags_warp; ++q) {
        m[q] = neg_inf; l[q] = 0.f;
    }

    // O accumulator: [qo_frags_warp=2][d_frags*N_REGS=32] = 64 f32
    float O_acc[K8_qo_frags_warp][K8_d_frags * N_REGS_PER_F32_ACCUM_FRAGMENT] = {};

    // -----------------------------------------------------------------------
    // Main KV loop
    // -----------------------------------------------------------------------
    for (int j = 0; j < args.n_KV_blocks; ++j) {

        // Load K smem (each warp: kv_rows_warp=16 rows)
        copy_tile_g2s<K8_kv_frags_warp, K8_D, false, true, value_t>(
            gmem_K + warp_id * K8_kv_rows_warp * stride,
            smem_K + warp_id * K8_kv_rows_warp * K8_D,
            stride, lane);
        asm volatile("cp.async.commit_group;");
        gmem_K += K8_Bc * stride;

        // Eager: start loading V while K is in flight
        copy_tile_g2s<K8_kv_frags_warp, K8_D, false, true, value_t>(
            gmem_V + warp_id * K8_kv_rows_warp * stride,
            smem_V + warp_id * K8_kv_rows_warp * K8_D,
            stride, lane);
        asm volatile("cp.async.commit_group;");
        gmem_V += K8_Bc * stride;

        // Wait for K
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        // ---- QK wgmma: S = Q × K^T ----
        float S_acc[K8_qo_frags_warp][K8_kv_frags * N_REGS_PER_F32_ACCUM_FRAGMENT];
        uint64_t k_desc = k8::make_smem_desc(smem_K);
        k8_wgmma_qk(S_acc, Q_regs, k_desc);

        // Wait for V
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // ---- Online softmax (exp2 path) ----
        float m_next[K8_qo_frags_warp];
        calc_row_max  (S_acc, m_next, m);
        scale_l_O<true>(m_next, m, l, O_acc, scale);
        exponentiate_tensor<true>(S_acc, m, scale);
        update_row_exp_sum(S_acc, l);

        // Convert S f32 → P fp16  (for wgmma A operand)
        uint32_t P_regs[K8_qo_frags_warp][K8_kv_frags];
        convert_to_16_bit_dtype<value_t>(S_acc, P_regs);

        // ---- PV wgmma: O += P × V ----
        uint64_t v_desc = k8::make_smem_desc(smem_V);
        k8_wgmma_pv(O_acc, P_regs, v_desc);
    }

    // -----------------------------------------------------------------------
    // Final normalization  O /= l
    // -----------------------------------------------------------------------
    final_softmax_normalization(O_acc, l);

    // -----------------------------------------------------------------------
    // Convert O f32 → fp16 and write smem → gmem
    // -----------------------------------------------------------------------
    uint32_t O_val[K8_qo_frags_warp][K8_d_frags];
    convert_to_16_bit_dtype<value_t>(O_acc, O_val);

    store_rf2smem<K8_qo_frags_warp, K8_d_frags, K8_D, false, value_t>(
        O_val,
        smem_Q + warp_id * K8_qo_rows_warp * K8_D,  // Q and O share smem
        lane);
    __syncwarp();

    copy_tile_s2g<K8_qo_frags_warp, K8_D, false, value_t>(
        gmem_O + warp_id * K8_qo_rows_warp * stride,
        smem_Q + warp_id * K8_qo_rows_warp * K8_D,
        stride, lane);
}

} // namespace flash
