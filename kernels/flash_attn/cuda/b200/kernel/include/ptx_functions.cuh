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

} // namespace flash
