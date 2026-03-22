"""
MXFP4 × MXFP4 GEMM in Triton for NVIDIA B200 (Blackwell SM100).

Problem
-------
Given:
  q_a     : uint8 [M, K//2]          — A packed 2×FP4 E2M1 per byte, row-major
  scale_a : uint8 (flat)             — A scales, E8M0, SWIZZLE_32_4_4 layout
  q_b     : uint8 [N, K//2]          — B packed 2×FP4 E2M1 per byte, row-major
  scale_b : uint8 (flat)             — B scales, E8M0, SWIZZLE_32_4_4 layout
  c       : float32 [M, N]           — output (pre-allocated)
  m, n, k : int                      — matrix dimensions; K and N divisible by 32

Compute:
  C[m,n] = Σ_k  dequant(A)[m,k]  *  dequant(B)[n,k]      (= A_dq @ B_dq^T)

Reference:
  F.scaled_mm(a_q, b_q.t(),
              scale_a=s_a, scale_recipe_a=BlockWise1x32, swizzle_a=SWIZZLE_32_4_4,
              scale_b=s_b, scale_recipe_b=BlockWise1x32, swizzle_b=SWIZZLE_32_4_4,
              output_dtype=torch.float32)

SWIZZLE_32_4_4 scale layout
---------------------------
For a matrix with M rows, K cols and 1×32 block scaling, the logical scale
tensor has shape [M, K//32].  SWIZZLE_32_4_4 tiles it as:
    physical shape : [⌈M/32⌉, ⌈K/128⌉, 32, 4]   (row-major 4-D)
    physical offset for logical index [m, kb]:
        row_tile = m // 32 ,  row_in = m % 32
        col_tile = kb // 4 ,  col_in = kb % 4
        offset   = row_tile * n_k_tiles * 128
                 + col_tile * 128
                 + row_in   * 4
                 + col_in
Each 32×4 = 128-byte tile fits one cache line, giving coalesced warp access.

FP4 E2M1 format
---------------
Bit layout: [sign(1) | exp(2) | mant(1)]   Bias = 1
  exp == 0  (subnormal) : (-1)^s * mant * 0.5
  exp != 0  (normal)    : (-1)^s * (1 + mant*0.5) * 2^(exp-1)
Values: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0

E8M0 format
-----------
8-bit unsigned exponent only:  scale = 2^(byte - 127)
(0xFF = NaN in spec; treated as 2^128 → inf here, safe for finite inputs.)

Kernel strategy
---------------
Outer tile : BLOCK_M × BLOCK_N in (M, N).
Inner loop : step BLOCK_K along K.  Within each BLOCK_K, iterate over
             BLOCK_K//32 scale groups of 32 K-elements (loop is unrolled).
Per group  : load 16 packed bytes → unpack to lo-nibbles [BM,16] and
             hi-nibbles [BM,16] (even/odd K respectively); same for B.
             Gather 1 E8M0 scale per row; scale A and B rows; then:
               acc += dot([BM,16], trans([BN,16]))   × 2 (lo + hi)
             Inner K dim = 16 satisfies Triton dot-product requirements.
"""

import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
#  FP4 E2M1 → float32 decoder
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _e2m1_to_f32(nibble):
    """
    Decode an E2M1 nibble (int32 tensor, values 0-15) to float32.
    Vectorised: operates element-wise on any shape.
    """
    sign  = (nibble >> 3) & 1          # sign bit
    exp2  = (nibble >> 1) & 3          # 2-bit exponent
    mant  = nibble & 1                 # 1-bit mantissa

    fexp  = exp2.to(tl.float32)
    fmant = mant.to(tl.float32)

    # Normal:    (1 + mant*0.5) * 2^(exp-1)
    # Subnormal: mant * 0.5        [covers 0.0 and ±0.5]
    normal    = (1.0 + fmant * 0.5) * tl.exp2(fexp - 1.0)
    subnormal = fmant * 0.5

    val = tl.where(exp2 == 0, subnormal, normal)
    val = tl.where(sign == 1, -val, val)
    return val                         # float32, same shape as nibble


# ──────────────────────────────────────────────────────────────────────────────
#  Main GEMM kernel
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def mxfp4_gemm_kernel(
    qa_ptr, scale_a_ptr,     # A data + A scales
    qb_ptr, scale_b_ptr,     # B data + B scales
    c_ptr,                   # FP32 output
    M, N, K,                 # matrix dimensions
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,   # must be ≥ 32 and a multiple of 32
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_base = pid_m * BLOCK_M
    n_base = pid_n * BLOCK_N

    # Row/col index vectors for this tile
    m_idx = m_base + tl.arange(0, BLOCK_M)          # [BM]
    n_idx = n_base + tl.arange(0, BLOCK_N)           # [BN]
    m_mask = m_idx < M
    n_mask = n_idx < N

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Derived constants
    K_bytes      = K // 2            # packed bytes per row in A/B
    # ⌈K/128⌉ — number of [32,4] scale tiles along K dimension
    n_k_sc_tiles = tl.cdiv(K, 128)

    # ── Precompute per-row SWIZZLE quantities (invariant across k-tiles) ──
    # A-side
    a_r_tile = m_idx >> 5            # m // 32   [BM]
    a_r_in   = m_idx & 31            # m %  32   [BM]
    # B-side
    b_r_tile = n_idx >> 5
    b_r_in   = n_idx & 31

    # ── Main K loop ───────────────────────────────────────────────────────
    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_base_val = k_tile * BLOCK_K

        # Inner loop over BLOCK_K/32 scale groups — unrolled by compiler
        # because BLOCK_K is tl.constexpr.
        for sub in range(BLOCK_K // 32):
            # ── Addressing for this 32-K-element scale group ──────────────
            k_lo  = k_base_val + sub * 32   # first logical K index
            k_byt = k_lo >> 1               # byte offset:  32 FP4 = 16 bytes
            kb    = k_lo >> 5               # scale-block index = k_lo // 32

            byte_range = k_byt + tl.arange(0, 16)   # [16] byte offsets

            # ── Load & unpack A bytes  [BM, 16] ──────────────────────────
            a_offs = m_idx[:, None] * K_bytes + byte_range[None, :]     # [BM,16]
            a_msk  = m_mask[:, None] & (byte_range[None, :] < K_bytes)
            a_raw  = tl.load(qa_ptr + a_offs, mask=a_msk, other=0).to(tl.int32)

            # lower nibble → even K-positions; upper → odd K-positions
            a_lo_f = _e2m1_to_f32(a_raw & 0x0F)           # [BM, 16]
            a_hi_f = _e2m1_to_f32((a_raw >> 4) & 0x0F)    # [BM, 16]

            # ── Load & unpack B bytes  [BN, 16] ──────────────────────────
            b_offs = n_idx[:, None] * K_bytes + byte_range[None, :]     # [BN,16]
            b_msk  = n_mask[:, None] & (byte_range[None, :] < K_bytes)
            b_raw  = tl.load(qb_ptr + b_offs, mask=b_msk, other=0).to(tl.int32)

            b_lo_f = _e2m1_to_f32(b_raw & 0x0F)
            b_hi_f = _e2m1_to_f32((b_raw >> 4) & 0x0F)

            # ── Load A scales — gather from SWIZZLE_32_4_4 layout ─────────
            # offset[m, kb] = row_tile*n_k_sc_tiles*128 + col_tile*128
            #                + row_in*4 + col_in
            col_tile = kb >> 2                        # kb // 4  (scalar: kb is const in unrolled sub)
            col_in   = kb & 3                         # kb %  4  (scalar)

            sa_off = (a_r_tile * (n_k_sc_tiles * 128)
                      + col_tile * 128
                      + a_r_in * 4
                      + col_in)                       # [BM] physical byte offsets
            sa_byte = tl.load(scale_a_ptr + sa_off, mask=m_mask, other=127).to(tl.int32)
            # E8M0:  scale = 2^(byte − 127)
            sa = tl.exp2(sa_byte.to(tl.float32) - 127.0)   # [BM]

            # ── Load B scales — gather from SWIZZLE_32_4_4 layout ─────────
            sb_off = (b_r_tile * (n_k_sc_tiles * 128)
                      + col_tile * 128
                      + b_r_in * 4
                      + col_in)                       # [BN]
            sb_byte = tl.load(scale_b_ptr + sb_off, mask=n_mask, other=127).to(tl.int32)
            sb = tl.exp2(sb_byte.to(tl.float32) - 127.0)   # [BN]

            # ── Scale A and B rows ────────────────────────────────────────
            a_lo_s = a_lo_f * sa[:, None]             # [BM, 16]
            a_hi_s = a_hi_f * sa[:, None]             # [BM, 16]
            b_lo_s = b_lo_f * sb[:, None]             # [BN, 16]
            b_hi_s = b_hi_f * sb[:, None]             # [BN, 16]

            # ── Dot-product accumulation ──────────────────────────────────
            # C += A_scaled @ B_scaled^T  decomposed into lo + hi halves:
            #   [BM,16] @ trans([BN,16]) = [BM,16] @ [16,BN] → [BM,BN]
            acc = tl.dot(a_lo_s, tl.trans(b_lo_s), acc=acc, input_precision="ieee")
            acc = tl.dot(a_hi_s, tl.trans(b_hi_s), acc=acc, input_precision="ieee")

    # ── Write C ──────────────────────────────────────────────────────────────
    c_offs = m_idx[:, None] * N + n_idx[None, :]
    c_msk  = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_offs, acc, mask=c_msk)


# ──────────────────────────────────────────────────────────────────────────────
#  Autotuned kernel wrapper (optional — comment out if warm-up latency matters)
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def mxfp4_gemm_kernel_autotuned(
    qa_ptr, scale_a_ptr,
    qb_ptr, scale_b_ptr,
    c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Autotuned version — identical body, wrapped for block-size search."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_base = pid_m * BLOCK_M
    n_base = pid_n * BLOCK_N

    m_idx = m_base + tl.arange(0, BLOCK_M)
    n_idx = n_base + tl.arange(0, BLOCK_N)
    m_mask = m_idx < M
    n_mask = n_idx < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_bytes      = K // 2
    n_k_sc_tiles = tl.cdiv(K, 128)

    a_r_tile = m_idx >> 5
    a_r_in   = m_idx & 31
    b_r_tile = n_idx >> 5
    b_r_in   = n_idx & 31

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_base_val = k_tile * BLOCK_K

        for sub in range(BLOCK_K // 32):
            k_lo  = k_base_val + sub * 32
            k_byt = k_lo >> 1
            kb    = k_lo >> 5

            byte_range = k_byt + tl.arange(0, 16)

            a_offs = m_idx[:, None] * K_bytes + byte_range[None, :]
            a_msk  = m_mask[:, None] & (byte_range[None, :] < K_bytes)
            a_raw  = tl.load(qa_ptr + a_offs, mask=a_msk, other=0).to(tl.int32)
            a_lo_f = _e2m1_to_f32(a_raw & 0x0F)
            a_hi_f = _e2m1_to_f32((a_raw >> 4) & 0x0F)

            b_offs = n_idx[:, None] * K_bytes + byte_range[None, :]
            b_msk  = n_mask[:, None] & (byte_range[None, :] < K_bytes)
            b_raw  = tl.load(qb_ptr + b_offs, mask=b_msk, other=0).to(tl.int32)
            b_lo_f = _e2m1_to_f32(b_raw & 0x0F)
            b_hi_f = _e2m1_to_f32((b_raw >> 4) & 0x0F)

            col_tile = kb >> 2
            col_in   = kb & 3

            sa_off  = a_r_tile * (n_k_sc_tiles * 128) + col_tile * 128 + a_r_in * 4 + col_in
            sa_byte = tl.load(scale_a_ptr + sa_off, mask=m_mask, other=127).to(tl.int32)
            sa      = tl.exp2(sa_byte.to(tl.float32) - 127.0)

            sb_off  = b_r_tile * (n_k_sc_tiles * 128) + col_tile * 128 + b_r_in * 4 + col_in
            sb_byte = tl.load(scale_b_ptr + sb_off, mask=n_mask, other=127).to(tl.int32)
            sb      = tl.exp2(sb_byte.to(tl.float32) - 127.0)

            a_lo_s = a_lo_f * sa[:, None]
            a_hi_s = a_hi_f * sa[:, None]
            b_lo_s = b_lo_f * sb[:, None]
            b_hi_s = b_hi_f * sb[:, None]

            acc = tl.dot(a_lo_s, tl.trans(b_lo_s), acc=acc, input_precision="ieee")
            acc = tl.dot(a_hi_s, tl.trans(b_hi_s), acc=acc, input_precision="ieee")

    c_offs = m_idx[:, None] * N + n_idx[None, :]
    c_msk  = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_offs, acc, mask=c_msk)


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point  (no  ops — pure Triton)
# ──────────────────────────────────────────────────────────────────────────────
def solution(q_a, scale_a, q_b, scale_b, c, m: int, n: int, k: int):
    """
    MXFP4 GEMM  C = dequant(A) @ dequant(B)^T   (FP32 output, no ).

    Parameters
    ----------
    q_a     : device uint8 tensor  [m, k//2]  — A packed FP4 E2M1 bytes
    scale_a : device uint8 tensor  (flat)     — A E8M0 scales, SWIZZLE_32_4_4
    q_b     : device uint8 tensor  [n, k//2]  — B packed FP4 E2M1 bytes
    scale_b : device uint8 tensor  (flat)     — B E8M0 scales, SWIZZLE_32_4_4
    c       : device float32 tensor [m, n]    — pre-allocated output
    m,n,k   : int                             — dims; K,N divisible by 32
    """
    # Fixed config (swap to autotuned variant for benchmarking)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))

    mxfp4_gemm_kernel[grid](
        q_a, scale_a,
        q_b, scale_b,
        c,
        m, n, k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    # ── To use the autotuned version instead, replace the call above with: ──
    # mxfp4_gemm_kernel_autotuned[grid](
    #     q_a, scale_a, q_b, scale_b, c, m, n, k,
    # )