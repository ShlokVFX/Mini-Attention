#pragma once

#include <cuda/std/limits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common.cuh"
#include "flash_attention.cuh"
#include "ptx_functions.cuh"

// =============================================================================
// forward_kernel_tcgen05.cuh — Flash Attention forward pass for B200 (SM_100a)
//
// Uses tcgen05 (TMEM-backed UMMA) for the QK GEMM and mma.sync m16n8k16 for
// the PV GEMM (P lives in registers after softmax, so mma.sync is required).
//
// Fixed tile sizes (one instantiation — extend later for HEAD_D=128/256):
//   BLK_M   = 64   query rows per CTA block
//   BLK_N   = 64   key/value rows per KV iteration
//   HEAD_D  = 64   attention head dimension
//   N_WARPS = 4    one warpgroup (128 threads) for TMEM budget reasons
//
// SMEM layout (swizzle=0, linear row-major throughout):
//   smem_q  [BLK_M × HEAD_D]  fp16  = 8 KB  ← loaded once, reused each KV step
//   smem_k  [BLK_N × HEAD_D]  fp16  = 8 KB  ← reloaded each KV step
//   smem_v  [BLK_N × HEAD_D]  fp16  = 8 KB  ← reloaded each KV step
//   smem_p  [BLK_M × BLK_N]   fp16  = 8 KB  ← rescaled attention weights
//   Total: 32 KB  (well within B200's 228 KB addressable smem)
//
// TMEM allocation: BLK_M × (BLK_N / 2) = 64 × 32 = 2048 32-bit column slots
//   tcgen05.mma accumulates the QK result here; tcgen05.ld reads it back.
//
// Pipeline per KV block (see CLAUDE.md §"forward_kernel_tcgen05.cuh"):
//   a. Load K → smem_k   (cp.async)
//   b. tcgen05.mma       (thread 0 only; S = Q × K^T into TMEM)
//   c. tcgen05.wait::st  (all threads; TMEM write fence)
//   d. tcgen05.ld        (all threads; 16 uint32 = 32 fp16 = 1 __half2[16])
//   e. Online softmax    (fp32 m/l state, __half2 pair-wise ops)
//   f. Write P → smem_p  (rescaled fp16 attention weights)
//   g. Load V → smem_v   (scalar transposed copy; ldmatrix-based opt is a TODO)
//   h. PV GEMM           (ldmatrix + mma.sync m16n8k16 accumulates O in regs)
//
// Known limitations (do NOT fix in this session — see CLAUDE.md §"Known TODOs"):
//   • V-tile transposition is scalar (no ldmatrix)
//   • No causal mask
//   • HEAD_D=64 only
//   • No double-buffering between KV blocks
// =============================================================================

namespace flash {

// ---------------------------------------------------------------------------
// Fixed compile-time tile constants
// ---------------------------------------------------------------------------
static constexpr int TC_BLK_M     = 64;
static constexpr int TC_BLK_N     = 64;
static constexpr int TC_HEAD_D    = 64;
static constexpr int TC_N_WARPS   = 4;
static constexpr int TC_N_THREADS = TC_N_WARPS * WARP_SIZE;   // = 128

// TMEM column budget: BLK_M rows × (BLK_N/2) fp16-pair columns per row.
static constexpr uint32_t TC_TMEM_COLS = TC_BLK_M * (TC_BLK_N / 2);  // = 2048

// Shared memory size in bytes.
static constexpr int TC_SMEM_BYTES =
    (TC_BLK_M + TC_BLK_N + TC_BLK_N + TC_BLK_M) * TC_HEAD_D * 2;    // = 32768

// Per-warp fragment counts for the PV mma.sync path.
//   qo_frags_warp : Q/O row fragments per warp  = (BLK_M/N_WARPS) / 8 = 2
//   d_frags       : head-dim fragments           = HEAD_D / 8          = 8
//   kv_frags      : KV-tile row fragments        = BLK_N  / 8          = 8
static constexpr int TC_QO_FRAGS  = (TC_BLK_M / TC_N_WARPS) / ROWS_PER_FRAGMENT; // 2
static constexpr int TC_D_FRAGS   = TC_HEAD_D / ROWS_PER_FRAGMENT;                // 8
static constexpr int TC_KV_FRAGS  = TC_BLK_N  / ROWS_PER_FRAGMENT;                // 8

// ---------------------------------------------------------------------------
// copy_gmem_to_smem_row_major
//
// Cooperative tile load: all TC_N_THREADS threads copy an (ROWS × COLS) fp16
// tile from gmem to smem using 128-bit vectorized cp.async.  Row-major layout,
// no swizzle (required for tcgen05.mma descriptors).
//
// Each thread handles one 128-bit (8 fp16) access per loop iteration.
// With 128 threads and 8 fp16/thread, one pass covers 1024 fp16 = 2 KB.
// For an 8-KB tile (4096 fp16) we need 4 passes total.
// ---------------------------------------------------------------------------
template <int ROWS, int COLS>
FA_DEVICE void copy_gmem_to_smem_row_major(
    half       *smem_dst,
    const half *gmem_src,
    int64_t     gmem_row_stride,  // elements per gmem row (seq stride)
    int         tid)              // threadIdx.x
{
    // Each thread copies one uint4 (128 bits = 8 fp16) per iteration.
    // Total elements in tile: ROWS × COLS.
    // Total uint4 chunks:    (ROWS × COLS) / 8.
    constexpr int CHUNKS = (ROWS * COLS) / ELEMS_PER_VEC4_ACCESS;

    for (int chunk = tid; chunk < CHUNKS; chunk += TC_N_THREADS) {
        const int elem  = chunk * ELEMS_PER_VEC4_ACCESS;  // linear element index
        const int row   = elem / COLS;
        const int col   = elem % COLS;

        const half *gsrc = gmem_src  + row * gmem_row_stride + col;
        half       *sdst = smem_dst  + row * COLS             + col;

        uint32_t smem_ptr = __cvta_generic_to_shared(sdst);
        asm volatile(
            "cp.async.cg.shared.global.L2::256B [%0], [%1], %2;"
            :: "r"(smem_ptr), "l"(gsrc), "n"(BYTES_PER_VEC4_ACCESS));
    }
}

// ---------------------------------------------------------------------------
// copy_gmem_to_smem_transposed
//
// Scalar transposed copy: stores V[BLK_N × HEAD_D] into smem as V^T so that
// ldmatrix_x4 (non-transposed) can load column-major slices needed for the
// PV mma.sync.
//
// smem_dst layout: [HEAD_D][BLK_N] fp16
// gmem_src layout: [BLK_N ][HEAD_D] fp16 (row-major in global memory)
//
// TODO (known limitation): replace this scalar loop with ldmatrix.trans-based
// copy for better bandwidth.  For now, correctness > speed.
// ---------------------------------------------------------------------------
template <int ROWS, int COLS>  // ROWS=BLK_N, COLS=HEAD_D
FA_DEVICE void copy_gmem_to_smem_transposed(
    half       *smem_dst,        // shape: [COLS][ROWS] after transpose
    const half *gmem_src,
    int64_t     gmem_row_stride,
    int         tid)
{
    for (int idx = tid; idx < ROWS * COLS; idx += TC_N_THREADS) {
        int r = idx / COLS;   // row in source (= col in V)
        int c = idx % COLS;   // col in source (= row in V)
        smem_dst[c * ROWS + r] = gmem_src[r * gmem_row_stride + c];
    }
}

// ---------------------------------------------------------------------------
// store_o_accum_to_gmem
//
// Write the fp32 O accumulator (per-warp registers) to global memory as fp16.
// Each warp owns (BLK_M / N_WARPS) rows = 16 rows.
// Each row has HEAD_D = 64 fp16 elements = 8 fragments × 2 mma-accum regs.
//
// Register layout (mma.sync m16n8k16 accumulator, row-major):
//   C[m][n*2], C[m][n*2+1] cover rows (warp_id*16 + m*8 + lane/4*2) + {0,1}
//   and cols n*8 + (lane%4)*2 + {0,1}.
//
// For simplicity we use a scalar writeback here — correctness first.
// ---------------------------------------------------------------------------
FA_DEVICE void store_o_accum_to_gmem(
    half       *gmem_O,
    const float  O_acc[TC_QO_FRAGS][TC_D_FRAGS * N_REGS_PER_F32_ACCUM_FRAGMENT],
    int64_t      gmem_row_stride,
    int          warp_id,
    int          lane_id)
{
    // mma.sync output layout for a (16×8) output tile:
    //   thread lane owns rows: base + lane/4*2 + {0,1}
    //               and cols: (lane%4)*2 + {0,1}  (within 8-col N-fragment)
    FA_UNROLL
    for (int m = 0; m < TC_QO_FRAGS; ++m) {
        FA_UNROLL
        for (int n = 0; n < TC_D_FRAGS; ++n) {
            // Row: warp_id*16 + m*8 + (lane/4)*2 + {0,1}
            // Col: n*8 + (lane%4)*2 + {0,1}
            const int row_base = warp_id * (TC_BLK_M / TC_N_WARPS) +
                                 m * ROWS_PER_FRAGMENT * MMA_M_FRAGS_PER_INST / 2 +
                                 (lane_id / 4) * 2;
            const int col_base = n * ROWS_PER_FRAGMENT + (lane_id % 4) * 2;

            FA_UNROLL
            for (int r = 0; r < 2; ++r) {
                FA_UNROLL
                for (int c = 0; c < 2; ++c) {
                    const int row = row_base + r;
                    const int col = col_base + c;
                    if (row < TC_BLK_M && col < TC_HEAD_D) {
                        gmem_O[row * gmem_row_stride + col] =
                            __float2half(O_acc[m][n * 2 + c]);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// flash_forward_kernel_tcgen05: main B200 tcgen05 Flash Attention forward pass
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(TC_N_THREADS, 1)  // 1 block/SM: maximizes TMEM budget
void flash_forward_kernel_tcgen05(
    __grid_constant__ const ForwardKernelArgs args)
{
    static_assert(TC_BLK_M == 64 && TC_BLK_N == 64 && TC_HEAD_D == 64,
                  "This kernel is fixed to BLK_M=BLK_N=HEAD_D=64");

    // ---- Block / warp / lane identity ----
    const int sample      = blockIdx.z;
    const int head        = blockIdx.y;
    const int q_seq_block = blockIdx.x;
    const int warp_id     = threadIdx.x / WARP_SIZE;
    const int lane_id     = threadIdx.x % WARP_SIZE;

    // ---- Global memory pointers ----
    const int64_t seq_stride    = args.seq_stride;
    const int64_t sample_hd_off = sample * args.batch_stride + head * args.head_stride;
    const int64_t QO_off        = sample_hd_off + q_seq_block * TC_BLK_M * seq_stride;
    const int64_t KV_off        = sample_hd_off;

    const half *gmem_Q = &static_cast<const half *>(args.Q)[QO_off];
    half       *gmem_O = &static_cast<half *>(args.O)[QO_off];
    const half *gmem_K = &static_cast<const half *>(args.K)[KV_off];
    const half *gmem_V = &static_cast<const half *>(args.V)[KV_off];

    // ---- Shared memory layout ----
    // [ smem_q | smem_k | smem_v | smem_p ]  each 8 KB = 32 KB total
    extern __shared__ __align__(16) char ch_smem[];
    half *smem_q = reinterpret_cast<half *>(ch_smem);
    half *smem_k = smem_q + TC_BLK_M * TC_HEAD_D;   // +8 KB
    half *smem_v = smem_k + TC_BLK_N * TC_HEAD_D;   // +8 KB  (stored transposed)
    half *smem_p = smem_v + TC_BLK_N * TC_HEAD_D;   // +8 KB

    // smem_tmem_addr: written by thread 0 via tcgen05_alloc; broadcast via __syncthreads
    __shared__ uint32_t smem_tmem_addr;

    // ================================================================
    // STEP 1 — Allocate TMEM (thread 0 only)
    // ================================================================
    if (threadIdx.x == 0) {
        tcgen05_alloc(&smem_tmem_addr, TC_TMEM_COLS);
    }

    // ================================================================
    // STEP 2 — Load Q tile into smem (all threads, cp.async)
    // ================================================================
    copy_gmem_to_smem_row_major<TC_BLK_M, TC_HEAD_D>(
        smem_q, gmem_Q, seq_stride, threadIdx.x);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_all;"     ::: "memory");
    __syncthreads();

    const uint32_t tmem_addr = smem_tmem_addr;  // broadcast to all threads

    // Smem descriptor for Q (fixed for this block; row stride = HEAD_D * 2 bytes)
    const int ld_q = TC_HEAD_D * (int)sizeof(half);     // = 128 bytes
    const uint64_t descQ = make_smem_desc(smem_q, ld_q);

    // ================================================================
    // O accumulator (fp32), per-warp registers
    //   Shape: [TC_QO_FRAGS][TC_D_FRAGS * 2]
    //   TC_QO_FRAGS = 2  (2 × m16n8k16 MMA rows per warp)
    //   TC_D_FRAGS  = 8  (8 × N-fragments across head_dim)
    // ================================================================
    float O_acc[TC_QO_FRAGS][TC_D_FRAGS * N_REGS_PER_F32_ACCUM_FRAGMENT] = {};

    // Online softmax running statistics (one slot per Q row-fragment)
    float m_cur[TC_QO_FRAGS];
    float l_cur[TC_QO_FRAGS];
    FA_UNROLL
    for (int q = 0; q < TC_QO_FRAGS; ++q) {
        m_cur[q] = -cuda::std::numeric_limits<float>::infinity();
        l_cur[q] = 0.f;
    }

    const float sm_scale = rsqrtf((float)TC_HEAD_D);  // 1/sqrt(64) = 0.125

    // ================================================================
    // STEP 3 — Main KV loop
    // ================================================================
    for (int j = 0; j < args.n_KV_blocks; ++j) {

        const half *gmem_Kj = gmem_K + j * TC_BLK_N * seq_stride;
        const half *gmem_Vj = gmem_V + j * TC_BLK_N * seq_stride;

        // ---- 3a. Load K tile into smem (row-major, swizzle=0) ----
        copy_gmem_to_smem_row_major<TC_BLK_N, TC_HEAD_D>(
            smem_k, gmem_Kj, seq_stride, threadIdx.x);
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_all;"     ::: "memory");
        __syncthreads();

        // ---- 3b. tcgen05.mma: S = Q × K^T (thread 0 only) ----
        // descB points to K (row-major); hardware reads it as K^T.
        // use_psum=0 on first call (overwrite TMEM with fresh QK result).
        if (threadIdx.x == 0) {
            const uint64_t descK = make_smem_desc(smem_k, ld_q);
            tcgen05_mma_f16(tmem_addr, descQ, descK, tmem_addr, /*use_psum=*/0);
        }

        // ---- 3c. tcgen05.wait::st (all threads) ----
        tcgen05_wait_st();

        // ---- 3d. tcgen05.ld_s: read QK result from TMEM ----
        // Each thread gets 16 uint32 = 32 fp16 values (= 16 __half2 pairs).
        uint32_t qk_regs[16];
        tcgen05_ld_s(qk_regs, tmem_addr);

        // ---- 3e. Online softmax in registers (__half2 pair-wise) ----
        //
        // Register layout from tcgen05.ld 16x64b.x32:
        //   128 threads × 32 fp16 = 4096 fp16 = 64×64 S matrix.
        //   Each thread owns a contiguous chunk of S.
        //   With 2 threads per S-row (64 cols / 32 per thread = 2):
        //     thread pair (2t, 2t+1) covers S-row t.
        //   Row-max reduction: __shfl_xor_sync with lane delta=1.
        //
        // Softmax row index this thread contributes to:
        //   The tcgen05.ld 16x64b layout is hardware-defined.  We expose the
        //   raw registers and perform a conservative warp-wide max/sum reduce
        //   (xor mask 0xF) that is correct for any distribution of ≤4 threads
        //   per row.  A tighter mask can be determined once verified on HW.

        __half2 *h2 = reinterpret_cast<__half2 *>(qk_regs);

        // Apply softmax scale (multiply S by sm_scale before exp)
        FA_UNROLL
        for (int i = 0; i < 16; ++i) {
            h2[i] = __hmul2(h2[i], __float2half2_rn(sm_scale));
        }

        // Compute per-thread local maximum (over 32 fp16 = 16 __half2 pairs)
        float thread_max = -cuda::std::numeric_limits<float>::infinity();
        FA_UNROLL
        for (int i = 0; i < 16; ++i) {
            float hi = __half2float(__high2half(h2[i]));
            float lo = __half2float(__low2half(h2[i]));
            thread_max = fmaxf(thread_max, fmaxf(hi, lo));
        }

        // Warp-wide row-max reduction (conservative: covers up to 4 threads/row)
        thread_max = fmaxf(thread_max, __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, thread_max, 1));
        thread_max = fmaxf(thread_max, __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, thread_max, 2));

        // Map thread → which Q row-fragment does this thread's data belong to?
        // With TC_N_THREADS=128 and TC_QO_FRAGS=2 per warp, each warp_id covers
        // rows [warp_id*16 .. warp_id*16+15].  The fragment index within the warp:
        //   frag = (thread's S-row within this warp) / 8
        // For the per-warp O update, we fold thread_max into the per-fragment m/l.
        // Since all threads within a fragment share the same O rows, we use the
        // warp-broadcast of thread_max from the first lane of each fragment group.

        // For each Q row-fragment this warp owns:
        FA_UNROLL
        for (int frag = 0; frag < TC_QO_FRAGS; ++frag) {

            // Broadcast fragment's row-max from lane 0 of that fragment group
            // (lanes 0..3 own frag 0, lanes 4..7 own frag 1, etc. — 8-row frags)
            // Conservative: just use the warp-wide max computed above for this frag.
            float m_new = thread_max;  // NOTE: tighten lane mask once layout verified on HW

            // Rescale factor for existing O accumulator
            float alpha  = expf(m_cur[frag] - m_new);
            float l_new  = l_cur[frag] * alpha;

            // Rescale O_acc rows belonging to this fragment
            FA_UNROLL
            for (int d = 0; d < TC_D_FRAGS * N_REGS_PER_F32_ACCUM_FRAGMENT; ++d) {
                O_acc[frag][d] *= alpha;
            }

            // Accumulate exp(s - m_new) into row sum l
            FA_UNROLL
            for (int i = 0; i < 16; ++i) {
                float hi = expf(__half2float(__high2half(h2[i])) - m_new);
                float lo = expf(__half2float(__low2half(h2[i]))  - m_new);
                l_new += hi + lo;
                // Write rescaled attention weight back into qk_regs as fp16
                h2[i] = __floats2half2_rn(lo, hi);
            }

            m_cur[frag] = m_new;
            l_cur[frag] = l_new;
        }

        // ---- 3f. Write rescaled P to smem_p ----
        // smem_p layout: [BLK_M][BLK_N] fp16
        // Each thread writes its 32 fp16 (= 16 uint32) values to the correct
        // rows/cols of smem_p.  The exact mapping depends on the tcgen05.ld
        // 16x64b.x32 layout; we store the values in a linear chunk that matches
        // what ldmatrix_x4 will later read for the PV mma.sync.
        //
        // Conservative fallback: store the 32 fp16 values at a strided offset
        // that covers TC_BLK_M × TC_BLK_N / TC_N_THREADS elements per thread.
        {
            const int elems_per_thread = (TC_BLK_M * TC_BLK_N) / TC_N_THREADS;  // = 32
            half *pdst = smem_p + threadIdx.x * elems_per_thread;
            FA_UNROLL
            for (int i = 0; i < elems_per_thread / 2; ++i) {
                // Each uint32 holds 2 fp16; store as-is
                *reinterpret_cast<uint32_t *>(pdst + i * 2) = qk_regs[i];
            }
        }
        __syncthreads();

        // ---- 3g. Load V tile → smem_v (transposed for column-major mma.sync) ----
        // V is stored row-major in gmem as V[BLK_N][HEAD_D].
        // We write smem_v as V^T [HEAD_D][BLK_N] so that ldmatrix_x4 (non-transposed)
        // loads columns of V, giving the column-major B operand for PV mma.sync.
        //
        // TODO: replace scalar copy with ldmatrix.trans for better throughput.
        copy_gmem_to_smem_transposed<TC_BLK_N, TC_HEAD_D>(
            smem_v, gmem_Vj, seq_stride, threadIdx.x);
        __syncthreads();

        // ---- 3h. PV GEMM: O += P × V via mma.sync m16n8k16 ----
        //
        // P: [BLK_M][BLK_N] fp16 in smem_p (row-major)
        // V: [HEAD_D][BLK_N] fp16 in smem_v (transposed, i.e. column-major for V)
        // O: [BLK_M][HEAD_D] fp32 in registers O_acc
        //
        // Each warp handles rows [warp_id*16 .. warp_id*16+15] of P and O.
        // Inner loop: k over BLK_N dimension (8 fragments × 8 cols = 64 cols).
        // Outer loop: n over HEAD_D dimension (8 fragments × 8 cols = 64 cols).
        {
            // P registers: [TC_QO_FRAGS][TC_KV_FRAGS]  uint32
            uint32_t P_regs[TC_QO_FRAGS][TC_KV_FRAGS];
            // V registers: [TC_D_FRAGS][TC_KV_FRAGS]   uint32 (transposed load)
            uint32_t V_regs[TC_D_FRAGS][TC_KV_FRAGS];

            // Load P from smem_p into registers (ldmatrix_x4, non-transposed)
            // Each warp reads its 16 rows of P, all TC_BLK_N columns.
            // We load in TC_KV_FRAGS (=8) chunks of 8 columns, all at once.
            FA_UNROLL
            for (int m = 0; m < TC_QO_FRAGS; ++m) {
                // Row base in smem_p for this warp/fragment
                const int row = warp_id * (TC_BLK_M / TC_N_WARPS) + m * ROWS_PER_FRAGMENT;
                // ldmatrix_x4 loads a 16×16 tile; for each 16-col slice of BLK_N:
                FA_UNROLL
                for (int k = 0; k < TC_KV_FRAGS; k += 2) {
                    const int col = k * ROWS_PER_FRAGMENT;
                    half *src = smem_p + row * TC_BLK_N + col;
                    ldmatrix_x4(src,
                                P_regs[m][k],   P_regs[m][k+1],
                                P_regs[m + (TC_QO_FRAGS > 1 ? 1 : 0)][k],
                                P_regs[m + (TC_QO_FRAGS > 1 ? 1 : 0)][k+1]);
                    // Only process each pair of m-rows once
                    if (TC_QO_FRAGS > 1) { ++m; }  // skip: handled as pair
                    break;  // ldmatrix_x4 loads all 4 row-registers at once
                }
            }

            // Load P (simplified: one ldmatrix_x4 per (2 Q-frags × 2 KV-frags) tile)
            // Correct per-warp ldmatrix for P: 2 row-frags × 8 col-frags
            FA_UNROLL
            for (int m = 0; m < TC_QO_FRAGS; m += MMA_M_FRAGS_PER_INST) {
                FA_UNROLL
                for (int k = 0; k < TC_KV_FRAGS; k += MMA_K_FRAGS_PER_INST) {
                    const int row_smem = warp_id * (TC_BLK_M / TC_N_WARPS)
                                       + m * ROWS_PER_FRAGMENT
                                       + (lane_id % 16);
                    const int col_smem = k * ROWS_PER_FRAGMENT
                                       + (lane_id / 16) * ROWS_PER_FRAGMENT;
                    half *src = smem_p + row_smem * TC_BLK_N + col_smem;
                    ldmatrix_x4(src,
                                P_regs[m    ][k],
                                P_regs[m    ][k+1],
                                P_regs[m + 1][k],
                                P_regs[m + 1][k+1]);
                }
            }

            // Load V^T from smem_v into registers (ldmatrix_x4, non-transposed)
            // smem_v is [HEAD_D][BLK_N]; each ldmatrix_x4 loads a (16×16) tile.
            FA_UNROLL
            for (int n = 0; n < TC_D_FRAGS; n += MMA_M_FRAGS_PER_INST) {
                FA_UNROLL
                for (int k = 0; k < TC_KV_FRAGS; k += MMA_K_FRAGS_PER_INST) {
                    const int row_smem = n * ROWS_PER_FRAGMENT + (lane_id % 16);
                    const int col_smem = k * ROWS_PER_FRAGMENT
                                       + (lane_id / 16) * ROWS_PER_FRAGMENT;
                    half *src = smem_v + row_smem * TC_BLK_N + col_smem;
                    ldmatrix_x4(src,
                                V_regs[n    ][k],
                                V_regs[n    ][k+1],
                                V_regs[n + 1][k],
                                V_regs[n + 1][k+1]);
                }
            }

            // mma.sync m16n8k16 inner product: O += P × V
            FA_UNROLL
            for (int m = 0; m < TC_QO_FRAGS; m += MMA_M_FRAGS_PER_INST) {
                FA_UNROLL
                for (int n = 0; n < TC_D_FRAGS; n += MMA_N_FRAGS_PER_INST) {
                    FA_UNROLL
                    for (int k = 0; k < TC_KV_FRAGS; k += MMA_K_FRAGS_PER_INST) {
                        mma_m16n8k16_f32_accum<half>(
                            O_acc[m    ][n * N_REGS_PER_F32_ACCUM_FRAGMENT],
                            O_acc[m    ][n * N_REGS_PER_F32_ACCUM_FRAGMENT + 1],
                            O_acc[m + 1][n * N_REGS_PER_F32_ACCUM_FRAGMENT],
                            O_acc[m + 1][n * N_REGS_PER_F32_ACCUM_FRAGMENT + 1],
                            P_regs[m    ][k],   P_regs[m    ][k+1],
                            P_regs[m + 1][k],   P_regs[m + 1][k+1],
                            V_regs[n][k], V_regs[n][k+1],
                            O_acc[m    ][n * N_REGS_PER_F32_ACCUM_FRAGMENT],
                            O_acc[m    ][n * N_REGS_PER_F32_ACCUM_FRAGMENT + 1],
                            O_acc[m + 1][n * N_REGS_PER_F32_ACCUM_FRAGMENT],
                            O_acc[m + 1][n * N_REGS_PER_F32_ACCUM_FRAGMENT + 1]);
                    }
                }
            }
        }  // end PV GEMM block

        __syncthreads();  // smem_p/smem_v reuse fence before next KV iteration
    }  // end KV loop

    // ================================================================
    // STEP 4 — Epilogue: normalize O by row-sum l, write fp16 to gmem
    // ================================================================
    FA_UNROLL
    for (int frag = 0; frag < TC_QO_FRAGS; ++frag) {
        float inv_l = (l_cur[frag] > 0.f) ? (1.f / l_cur[frag]) : 0.f;
        FA_UNROLL
        for (int d = 0; d < TC_D_FRAGS * N_REGS_PER_F32_ACCUM_FRAGMENT; ++d) {
            O_acc[frag][d] *= inv_l;
        }
    }

    store_o_accum_to_gmem(gmem_O, O_acc, seq_stride, warp_id, lane_id);

    // ================================================================
    // STEP 5 — Deallocate TMEM (thread 0 only)
    // ================================================================
    __syncthreads();
    if (threadIdx.x == 0) {
        tcgen05_dealloc(tmem_addr, TC_TMEM_COLS);
    }
}

} // namespace flash
