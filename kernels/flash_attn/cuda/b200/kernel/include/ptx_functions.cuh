#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "common.cuh"

namespace flash {

// ---------------------------------------------------------------------------
// cp.async: asynchronous global-to-shared copy (bypasses L1 cache)
// B200 note: same L2::256B hint as SM_120 — Blackwell (both consumer and
//   datacenter) uses 256-byte physical L2 sectors.
// ---------------------------------------------------------------------------

template <bool async>
FA_DEVICE void cp_async_commit() {
    if constexpr (async) {
        asm volatile("cp.async.commit_group;");
    }
}

template <int ngroups, bool async>
FA_DEVICE void cp_async_wait() {
    if constexpr (async) {
        asm volatile("cp.async.wait_group %0;" ::"n"(ngroups));
    }
}

template <bool async>
FA_DEVICE_CONSTEXPR void cp_async_commit_and_wait_all() {
    if constexpr (async) {
        cp_async_commit<async>();
        cp_async_wait<0, async>();
    }
}

// 128-bit async gmem → smem copy with L2::256B eviction hint (B200 optimal).
template <int size, typename T>
FA_DEVICE void cp_async(T *smem_to, T *gmem_from) {
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_to);
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;"
                 :
                 : "r"(smem_ptr), "l"(gmem_from), "n"(size));
}

// ---------------------------------------------------------------------------
// ldmatrix: load matrix tiles from smem into registers (mma.sync path)
// ---------------------------------------------------------------------------

template <typename T>
FA_DEVICE void ldmatrix_x4(T *load_from, uint32_t &a1, uint32_t &a2,
                           uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16"
                 " {%0, %1, %2, %3}, [%4];"
                 : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                 : "r"(smem_ptr));
}

template <typename T>
FA_DEVICE void ldmatrix_x4_transpose(T *load_from, uint32_t &a1, uint32_t &a2,
                                     uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16"
                 " {%0, %1, %2, %3}, [%4];"
                 : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
                 : "r"(smem_ptr));
}

// ---------------------------------------------------------------------------
// mma.sync.m16n8k16: warp-level matrix multiply-accumulate (mma.sync path)
// ---------------------------------------------------------------------------
template <typename value_t>
FA_DEVICE void
mma_m16n8k16_f32_accum(float &d1, float &d2, float &d3, float &d4,
                       uint32_t const &a1, uint32_t const &a2,
                       uint32_t const &a3, uint32_t const &a4,
                       uint32_t const &b1, uint32_t const &b2,
                       float const &c1, float const &c2,
                       float const &c3, float const &c4) {
    static_assert(std::is_same_v<value_t, half> ||
                      std::is_same_v<value_t, nv_bfloat16>,
                  "value_t must be either half or nv_bfloat16");

    if constexpr (std::is_same_v<value_t, half>) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            " { %0,  %1,  %2,  %3 }, "
            " { %4,  %5,  %6,  %7 }, "
            " { %8,  %9 }, "
            " { %10, %11, %12, %13 }; "
            : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
            : "r"(a1), "r"(a2), "r"(a3), "r"(a4),
              "r"(b1), "r"(b2),
              "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            " { %0,  %1,  %2,  %3 }, "
            " { %4,  %5,  %6,  %7 }, "
            " { %8,  %9 }, "
            " { %10, %11, %12, %13 }; "
            : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
            : "r"(a1), "r"(a2), "r"(a3), "r"(a4),
              "r"(b1), "r"(b2),
              "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    }
}

// ===========================================================================
// B200 / SM_100 WGMMA — Warpgroup Matrix Multiply-Accumulate
//
// wgmma.mma_async reads A directly from shared memory (no ldmatrix needed).
// All 128 threads in the warpgroup issue the same instruction; the hardware
// dispatches the correct smem rows to each warp automatically.
//
// D += A × B  (f32 accumulator, f16/bf16 inputs)
//
// Output register layout per thread (same as stacked mma.sync m16n8k16):
//   Thread t in warpgroup covers 2 output rows per M-fragment:
//     row_base = (t % 4) * 2 + (t / 32) * 16
//   and N/4 output fp32 values (2 per 8-column N-tile).
// ===========================================================================

// ---------------------------------------------------------------------------
// build_smem_desc: encode a 64-bit smem descriptor for wgmma operands.
//   smem_ptr  — __shared__ address (result of __cvta_generic_to_shared)
//   ld_bytes  — leading dimension in bytes (row stride = cols * sizeof(elem))
//
// Descriptor layout (PTX ISA 8.x):
//   bits  0..13  smem_addr >> 4 (bits [17:4] of raw smem address)
//   bits 16..29  ld_bytes / 16  (leading dimension in 16-byte units)
//   bits 32..45  ld_bytes / 16  (base stride = same for 2-D tiles)
//   bits 62..63  swizzle mode:  3 = no interleave (linear), 0 = 128-B XOR
//
// We always use swizzle=3 here (no XOR swizzle) because the PTX swizzle
// mode must match how the tile was laid out with cp.async.  The mma.sync
// kernel path uses XOR swizzle (swizzling.cuh) while the WGMMA path uses
// a linear layout to keep the descriptor simple.
// ---------------------------------------------------------------------------
FA_DEVICE uint64_t build_smem_desc(const void *smem_ptr, int ld_bytes) {
    uint32_t addr32 = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc   = 0;
    desc |= (uint64_t)((addr32 >> 4) & 0x3FFF);          // bits  0..13
    desc |= (uint64_t)(ld_bytes / 16) << 16;              // bits 16..29
    desc |= (uint64_t)(ld_bytes / 16) << 32;              // bits 32..45
    desc |= (uint64_t)3ULL          << 62;                // bits 62..63: no swizzle
    return desc;
}

// ---------------------------------------------------------------------------
// wgmma_fence: issue a fence before issuing wgmma instructions.
// Required whenever threads modify accumulator registers (e.g. set to zero)
// and then switch to wgmma asynchronous execution.
// ---------------------------------------------------------------------------
FA_DEVICE void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

// ---------------------------------------------------------------------------
// wgmma_commit: mark the end of one wgmma instruction group.
// ---------------------------------------------------------------------------
FA_DEVICE void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

// ---------------------------------------------------------------------------
// wgmma_wait<N>: wait until ≤ N wgmma groups are still in flight.
// wgmma_wait<0>: wait for ALL wgmma instructions to complete.
// ---------------------------------------------------------------------------
template <int n_in_flight>
FA_DEVICE void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(n_in_flight) : "memory");
}

// ---------------------------------------------------------------------------
// wgmma_m64n64k16: warpgroup MMA — D(64×64) += A(64×16) × B(16×64)
//   32 fp32 output registers per thread.
//   scaleD=1 → D += A*B  (accumulate)
//   scaleD=0 → D  = A*B  (overwrite)
// ---------------------------------------------------------------------------
FA_DEVICE void
wgmma_m64n64k16_f16(float (&d)[32], uint64_t descA, uint64_t descB, int scaleD) {
    if (scaleD) {
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
            "%32, %33, 1, 1, 1, 0, 0;"
            : "=f"(d[0]),  "=f"(d[1]),  "=f"(d[2]),  "=f"(d[3]),
              "=f"(d[4]),  "=f"(d[5]),  "=f"(d[6]),  "=f"(d[7]),
              "=f"(d[8]),  "=f"(d[9]),  "=f"(d[10]), "=f"(d[11]),
              "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
              "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
              "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
              "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]),
              "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
            : "l"(descA), "l"(descB));
    } else {
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
            "%32, %33, 0, 1, 1, 0, 0;"
            : "=f"(d[0]),  "=f"(d[1]),  "=f"(d[2]),  "=f"(d[3]),
              "=f"(d[4]),  "=f"(d[5]),  "=f"(d[6]),  "=f"(d[7]),
              "=f"(d[8]),  "=f"(d[9]),  "=f"(d[10]), "=f"(d[11]),
              "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
              "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
              "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
              "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]),
              "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
            : "l"(descA), "l"(descB));
    }
}

// ---------------------------------------------------------------------------
// wgmma_m64n128k16: warpgroup MMA — D(64×128) += A(64×16) × B(16×128)
//   64 fp32 output registers per thread.
// ---------------------------------------------------------------------------
FA_DEVICE void
wgmma_m64n128k16_f16(float (&d)[64], uint64_t descA, uint64_t descB, int scaleD) {
    if (scaleD) {
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
            "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
            "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
            "%64, %65, 1, 1, 1, 0, 0;"
            : "=f"(d[0]),  "=f"(d[1]),  "=f"(d[2]),  "=f"(d[3]),
              "=f"(d[4]),  "=f"(d[5]),  "=f"(d[6]),  "=f"(d[7]),
              "=f"(d[8]),  "=f"(d[9]),  "=f"(d[10]), "=f"(d[11]),
              "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
              "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
              "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
              "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]),
              "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]),
              "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]),
              "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]),
              "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]),
              "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]),
              "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]),
              "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]),
              "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]),
              "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
            : "l"(descA), "l"(descB));
    } else {
        asm volatile(
            "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
            "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
            "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
            "%64, %65, 0, 1, 1, 0, 0;"
            : "=f"(d[0]),  "=f"(d[1]),  "=f"(d[2]),  "=f"(d[3]),
              "=f"(d[4]),  "=f"(d[5]),  "=f"(d[6]),  "=f"(d[7]),
              "=f"(d[8]),  "=f"(d[9]),  "=f"(d[10]), "=f"(d[11]),
              "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
              "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
              "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
              "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]),
              "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]),
              "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]),
              "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]),
              "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]),
              "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]),
              "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]),
              "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]),
              "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]),
              "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
            : "l"(descA), "l"(descB));
    }
}


// ===========================================================================
// B200 / SM_100a tcgen05 — Tensor Core GENerations 05 (TMEM-backed UMMA)
//
// tcgen05 is the SM_100a successor to wgmma (SM_90a).  Unlike wgmma, which
// requires all 128 warpgroup threads to issue the same instruction, tcgen05
// decouples the MMA launch from the result read:
//
//   1. ONE thread per CTA calls tcgen05.alloc → receives a TMEM base address.
//   2. ONE thread per CTA calls tcgen05.mma   → fires the UMMA instruction.
//   3. ALL threads call tcgen05.wait::st       → barrier; TMEM writes visible.
//   4. ALL threads call tcgen05.ld             → each thread reads its slice.
//   5. ONE thread per CTA calls tcgen05.dealloc→ releases TMEM.
//
// SMEM descriptor rule for B200:
//   Always build descriptors with swizzle=0 (bits[63:62]=0b00, no XOR).
//   The wgmma-era build_smem_desc sets bits[63:62]=0b11 (128B swizzle) which
//   is INCOMPATIBLE with tcgen05.mma and causes silent wrong results.
//   Use make_smem_desc() (below) which hard-codes swizzle=0.
//
// TMEM column sizing:
//   tcgen05.alloc takes ncols in 32-bit word units.
//   For a BLK_M × BLK_N fp16 output tile:
//     ncols = BLK_M × (BLK_N / 2)   (each col = one fp16 pair = 1 uint32)
// ===========================================================================

// ---------------------------------------------------------------------------
// make_smem_desc: build a 64-bit smem matrix descriptor for tcgen05.mma.
//   smem_ptr  — __shared__ base address of the matrix tile
//   ld_bytes  — leading dimension in bytes (= cols * sizeof(elem))
//
//   Descriptor bit layout (PTX ISA 8.7 §9.7.14.5):
//     bits  0..13  smem address >> 4
//     bits 16..29  ld_bytes / 16  (leading-dim in 16-B units)
//     bits 32..45  ld_bytes / 16  (stride; same value for simple 2-D tiles)
//     bits 62..63  swizzle mode: 0b00 = no swizzle  ← REQUIRED for tcgen05
//
//   DO NOT use build_smem_desc() here — it sets bits[63:62]=0b11 (128B XOR
//   swizzle) which is wrong for tcgen05.mma on B200.
// ---------------------------------------------------------------------------
FA_DEVICE uint64_t make_smem_desc(const void *smem_ptr, int ld_bytes) {
    uint32_t addr32 = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc   = 0;
    desc |= (uint64_t)((addr32 >> 4) & 0x3FFF);          // bits  0..13
    desc |= (uint64_t)(ld_bytes / 16) << 16;              // bits 16..29
    desc |= (uint64_t)(ld_bytes / 16) << 32;              // bits 32..45
    // bits 62..63 intentionally 0: no swizzle
    return desc;
}

// ---------------------------------------------------------------------------
// tcgen05_alloc: allocate TMEM columns for one CTA.
//
//   smem_out  — pointer into __shared__ memory; the allocated TMEM base
//               address is written here by the hardware.
//   ncols     — number of 32-bit column slots to reserve
//               (= BLK_M × (BLK_N / 2) for a BLK_M×BLK_N fp16 accumulator)
//
//   MUST be called by exactly one thread per CTA (threadIdx.x == 0).
//   Followed by __syncthreads() so all threads see the written address.
// ---------------------------------------------------------------------------
FA_DEVICE void tcgen05_alloc(uint32_t *smem_out, uint32_t ncols) {
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_out);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned [%0], %1;"
        :: "r"(smem_ptr), "r"(ncols) : "memory");
}

// ---------------------------------------------------------------------------
// tcgen05_dealloc: release previously allocated TMEM columns.
//
//   tmem_addr — TMEM base address returned by tcgen05_alloc
//   ncols     — same value passed to tcgen05_alloc
//
//   MUST be called by exactly one thread per CTA (threadIdx.x == 0).
//   Call after all tcgen05.ld reads are complete.
// ---------------------------------------------------------------------------
FA_DEVICE void tcgen05_dealloc(uint32_t tmem_addr, uint32_t ncols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned [%0], %1;"
        :: "r"(tmem_addr), "r"(ncols) : "memory");
}

// ---------------------------------------------------------------------------
// tcgen05_mma_f16: launch TMEM-backed UMMA.
//   Computes:  D[BLK_M × BLK_N] = A[BLK_M × HEAD_D] × B[BLK_N × HEAD_D]^T
//              (+ accumulate if use_psum=1)
//
//   tmem_d   — TMEM address of the output accumulator D  (same as tmem_c for
//               in-place accumulate; pass tmem_d == tmem_c when use_psum=1)
//   descA    — 64-bit smem descriptor for A (Q tile, row-major, swizzle=0)
//   descB    — 64-bit smem descriptor for B (K tile, row-major; hardware
//               reads B column-major, i.e. computes A × B^T automatically)
//   tmem_c   — TMEM address of the input accumulator C
//   use_psum — 0 = overwrite D with A×B; 1 = accumulate D += A×B
//
//   MUST be called by exactly one thread per CTA (threadIdx.x == 0).
//   After this call, issue tcgen05_wait_st() before any tcgen05_ld_s().
// ---------------------------------------------------------------------------
FA_DEVICE void tcgen05_mma_f16(uint32_t tmem_d, uint64_t descA, uint64_t descB,
                                uint32_t tmem_c, int use_psum) {
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, [%3], %4, 0, 0;"
        :: "r"(tmem_d), "l"(descA), "l"(descB), "r"(tmem_c), "r"(use_psum)
        : "memory");
}

// ---------------------------------------------------------------------------
// tcgen05_wait_st: CTA-wide barrier — wait until all tcgen05.mma TMEM writes
// are complete and visible to subsequent tcgen05.ld instructions.
//
//   Call from ALL threads after tcgen05_mma_f16, before tcgen05_ld_s.
//   This is the tcgen05 equivalent of wgmma_wait<0>().
// ---------------------------------------------------------------------------
FA_DEVICE void tcgen05_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// ---------------------------------------------------------------------------
// tcgen05_ld_s: load this thread's slice of the TMEM QK result.
//
//   Variant: 16x64b.x32 — loads 16 × uint32 registers per thread.
//   Interpretation: 16 uint32 = 32 fp16 values = 16 __half2 pairs.
//   For BLK_M=64, BLK_N=64, 128-thread CTA:
//     64 × 64 fp16 / 128 threads = 32 fp16 per thread = 16 uint32 ✓
//
//   regs[16] — output: this thread's 16 uint32 registers from TMEM
//   tmem_addr — TMEM base address from tcgen05_alloc
//
//   Called by ALL threads. Must be preceded by tcgen05_wait_st().
//   Reinterpret regs as __half2[16] for pair-wise softmax arithmetic.
// ---------------------------------------------------------------------------
FA_DEVICE void tcgen05_ld_s(uint32_t (&regs)[16], uint32_t tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x64b.x32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(regs[0]),  "=r"(regs[1]),  "=r"(regs[2]),  "=r"(regs[3]),
          "=r"(regs[4]),  "=r"(regs[5]),  "=r"(regs[6]),  "=r"(regs[7]),
          "=r"(regs[8]),  "=r"(regs[9]),  "=r"(regs[10]), "=r"(regs[11]),
          "=r"(regs[12]), "=r"(regs[13]), "=r"(regs[14]), "=r"(regs[15])
        : "r"(tmem_addr));
}

} // namespace flash
