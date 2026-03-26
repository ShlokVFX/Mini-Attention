#pragma once
// =============================================================================
// wgmma_ptx.cuh — SM90/SM120 warpgroup MMA (wgmma.mma_async) PTX wrappers
//
// Fixed shapes for K8 FlashAttention:
//   QK GEMM : wgmma.m64n64k16.f32.f16.f16  (transB=1: K stored as Bc×d_head)
//   PV GEMM : wgmma.m64n128k16.f32.f16.f16 (transB=0: V stored as Bc×d_head)
//
// One warpgroup = 4 warps = 128 threads per CTA.
// A operand comes from registers (same layout as ldmatrix_x4 / mma.sync).
// B operand comes from shared memory via 64-bit matrix descriptor.
// =============================================================================

#include <stdint.h>
#include <cuda_fp16.h>

namespace flash {
namespace k8 {

// ---------------------------------------------------------------------------
// Smem matrix descriptor for wgmma B operand.
//
// Descriptor bit layout (PTX ISA 8.7):
//   bits [13: 0]  stride_dim        in 16-byte units  (= row_stride_bytes / 16)
//   bits [29:16]  leading_dim_offset in 16-byte units  (= 0 for no swizzle)
//   bits [45:32]  start_address      in 16-byte units  (= smem_ptr >> 4)
//   bits [63:62]  swizzle_mode       (0 = no swizzle)
//
// For all K8 tiles: d_head=128, fp16 → row_stride = 256 bytes → stride_field=16.
// ---------------------------------------------------------------------------
__forceinline__ __device__
uint64_t make_smem_desc(const void* smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= 16ULL;                          // stride_dim = 256B / 16 = 16
    desc |= (uint64_t)(addr >> 4) << 32;   // start_address
    // leading_dim_offset = 0, swizzle = 0
    return desc;
}

// Advance K descriptor by one k16 step.
// K is (Bc, d_head) row-major; transB=1 so wgmma reads columns.
// k16 step = 16 columns × 2 bytes = 32 bytes → base += 32/16 = 2.
__forceinline__ __device__
uint64_t k_desc_advance(uint64_t desc) {
    return desc + (2ULL << 32);
}

// Advance V descriptor by one k16 step.
// V is (Bc, d_head) row-major; transB=0 so wgmma reads rows.
// k16 step = 16 rows × 256 bytes = 4096 bytes → base += 4096/16 = 256.
__forceinline__ __device__
uint64_t v_desc_advance(uint64_t desc) {
    return desc + (256ULL << 32);
}

// ---------------------------------------------------------------------------
// wgmma fence / commit / wait wrappers
// ---------------------------------------------------------------------------
__forceinline__ __device__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n");
}
__forceinline__ __device__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n");
}
// Wait for all prior wgmma groups to finish (N=0 means wait for all).
template <int N = 0>
__forceinline__ __device__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N));
}

// ---------------------------------------------------------------------------
// wgmma_qk: wgmma.m64n64k16.f32.f16.f16 — A from RF, B from smem, transB=1
//
// ScaleD=0: d[i]  = A*B        (initialize accumulators)
// ScaleD=1: d[i] += A*B        (accumulate)
//
// a0..a3  — 4 uint32 registers for the 64×16 A slice (from ldmatrix_x4)
// b_desc  — smem descriptor for the (Bc, 16) B slice (transB=1 → reads K^T)
// d[32]   — 32 f32 accumulator registers per thread (m64n64 / 128 threads)
// ---------------------------------------------------------------------------
template <int ScaleD>
__forceinline__ __device__ void
wgmma_qk(float (&d)[32],
          uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
          uint64_t b_desc)
{
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        " %8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,"
        " %24,%25,%26,%27,%28,%29,%30,%31},"
        "{%32,%33,%34,%35},"
        "%36,"
        "%37,1,1,1;\n"
        : "=f"(d[ 0]),"=f"(d[ 1]),"=f"(d[ 2]),"=f"(d[ 3]),
          "=f"(d[ 4]),"=f"(d[ 5]),"=f"(d[ 6]),"=f"(d[ 7]),
          "=f"(d[ 8]),"=f"(d[ 9]),"=f"(d[10]),"=f"(d[11]),
          "=f"(d[12]),"=f"(d[13]),"=f"(d[14]),"=f"(d[15]),
          "=f"(d[16]),"=f"(d[17]),"=f"(d[18]),"=f"(d[19]),
          "=f"(d[20]),"=f"(d[21]),"=f"(d[22]),"=f"(d[23]),
          "=f"(d[24]),"=f"(d[25]),"=f"(d[26]),"=f"(d[27]),
          "=f"(d[28]),"=f"(d[29]),"=f"(d[30]),"=f"(d[31])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "l"(b_desc),
          "n"(ScaleD)
    );
}

// ---------------------------------------------------------------------------
// wgmma_pv: wgmma.m64n128k16.f32.f16.f16 — A from RF, B from smem, transB=0
//
// d[64]   — 64 f32 accumulator registers per thread (m64n128 / 128 threads)
// ---------------------------------------------------------------------------
template <int ScaleD>
__forceinline__ __device__ void
wgmma_pv(float (&d)[64],
          uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
          uint64_t b_desc)
{
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,"
        " %8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,"
        " %24,%25,%26,%27,%28,%29,%30,%31,"
        " %32,%33,%34,%35,%36,%37,%38,%39,"
        " %40,%41,%42,%43,%44,%45,%46,%47,"
        " %48,%49,%50,%51,%52,%53,%54,%55,"
        " %56,%57,%58,%59,%60,%61,%62,%63},"
        "{%64,%65,%66,%67},"
        "%68,"
        "%69,1,1,0;\n"
        : "=f"(d[ 0]),"=f"(d[ 1]),"=f"(d[ 2]),"=f"(d[ 3]),
          "=f"(d[ 4]),"=f"(d[ 5]),"=f"(d[ 6]),"=f"(d[ 7]),
          "=f"(d[ 8]),"=f"(d[ 9]),"=f"(d[10]),"=f"(d[11]),
          "=f"(d[12]),"=f"(d[13]),"=f"(d[14]),"=f"(d[15]),
          "=f"(d[16]),"=f"(d[17]),"=f"(d[18]),"=f"(d[19]),
          "=f"(d[20]),"=f"(d[21]),"=f"(d[22]),"=f"(d[23]),
          "=f"(d[24]),"=f"(d[25]),"=f"(d[26]),"=f"(d[27]),
          "=f"(d[28]),"=f"(d[29]),"=f"(d[30]),"=f"(d[31]),
          "=f"(d[32]),"=f"(d[33]),"=f"(d[34]),"=f"(d[35]),
          "=f"(d[36]),"=f"(d[37]),"=f"(d[38]),"=f"(d[39]),
          "=f"(d[40]),"=f"(d[41]),"=f"(d[42]),"=f"(d[43]),
          "=f"(d[44]),"=f"(d[45]),"=f"(d[46]),"=f"(d[47]),
          "=f"(d[48]),"=f"(d[49]),"=f"(d[50]),"=f"(d[51]),
          "=f"(d[52]),"=f"(d[53]),"=f"(d[54]),"=f"(d[55]),
          "=f"(d[56]),"=f"(d[57]),"=f"(d[58]),"=f"(d[59]),
          "=f"(d[60]),"=f"(d[61]),"=f"(d[62]),"=f"(d[63])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "l"(b_desc),
          "n"(ScaleD)
    );
}

} // namespace k8
} // namespace flash
