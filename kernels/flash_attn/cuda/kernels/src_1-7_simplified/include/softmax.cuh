#pragma once

#include "common.cuh"

// =============================================================================
// softmax.cuh — online softmax operations (all register-level, no smem)
//
// Each function operates on per-warp accumulator arrays.
// One "row fragment" corresponds to ROWS_PER_FRAGMENT (= 8) rows of the Q tile
// owned by this warp.
// =============================================================================

namespace flash {

// Scale every element of the S accumulator by softmax_scale.
// Used when NOT in optimized_softmax mode (exp2 path).
template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
scale_S_accum(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
              const accum_t &softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        FA_UNROLL
        for (int k = 0; k < KV_accum_fragments; ++k)
            S_accum[q][k] *= softmax_scale;
    }
}

// Compute new row maximums m_next[q] = max(m_cur[q], max over S row q).
// Reduces across the 4 threads that share a row via warp shuffle.
template <int QO_fragments, int KV_accum_fragments, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
calc_row_max(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
             accum_t (&m_next)[QO_fragments],
             accum_t (&m_cur)[QO_fragments]) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        m_next[q] = m_cur[q];
        FA_UNROLL
        for (int k = 0; k < KV_accum_fragments; ++k)
            m_next[q] = max(m_next[q], S_accum[q][k]);

        // Reduce across the 4 threads in the same row (shuffle-XOR tree)
        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 2), m_next[q]);
        m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 1), m_next[q]);
    }
}

// Rescale the running row sum l and output accumulator O when the row max changes.
// Also updates m_cur = m_next.
template <bool optimized_softmax, int QO_fragments, int d_head_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
scale_l_O(accum_t (&m_next)[QO_fragments], accum_t (&m_cur)[QO_fragments],
          accum_t (&l)[QO_fragments],
          accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
          accum_t softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        accum_t scale;
        if constexpr (optimized_softmax)
            scale = exp2f((m_cur[q] - m_next[q]) * softmax_scale);
        else
            scale = expf(m_cur[q] - m_next[q]);

        m_cur[q] = m_next[q];
        l[q] *= scale;
        FA_UNROLL
        for (int d = 0; d < d_head_accum_fragments; ++d)
            O_accum[q][d] *= scale;
    }
}

// Replace each element of S with exp(s - m)  (or exp2 in optimized mode).
template <bool optimized_softmax, int QO_fragments, int KV_accum_fragments,
          typename accum_t = float>
FA_DEVICE_CONSTEXPR void
exponentiate_tensor(accum_t (&S_accum)[QO_fragments][KV_accum_fragments],
                    accum_t (&m)[QO_fragments], accum_t softmax_scale) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        accum_t max_scaled;
        if constexpr (optimized_softmax)
            max_scaled = m[q] * softmax_scale;

        FA_UNROLL
        for (int k = 0; k < KV_accum_fragments; ++k) {
            if constexpr (optimized_softmax)
                S_accum[q][k] = exp2f(S_accum[q][k] * softmax_scale - max_scaled);
            else
                S_accum[q][k] = expf(S_accum[q][k] - m[q]);
        }
    }
}

// Accumulate row sums into l[] (sum over the P tile columns).
template <int QO_fragments, int d_head_accum_fragments, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
update_row_exp_sum(accum_t (&P_accum)[QO_fragments][d_head_accum_fragments],
                   accum_t (&l)[QO_fragments]) {
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        FA_UNROLL
        for (int d = 0; d < d_head_accum_fragments; ++d)
            l[q] += P_accum[q][d];
    }
}

// Final normalization: O /= l.
// First completes the warp-level reduction of l, then divides every O element.
//
// SM120 optimization: pre-compute 1/l[q] once per row-fragment and replace
//   d_head_accum_fragments divisions with a single fast reciprocal followed by
//   multiplications.  Division maps to rcp+mul on the hardware anyway, but the
//   compiler cannot hoist it across the inner loop automatically in all cases.
//   With --use_fast_math __frcp_rn() is a single FMUL instruction latency ~4
//   cycles, versus a full IEEE divide (~20 cycles).  For d_head=128 and
//   qo_frags_warp=1 this saves ~15 division instructions per warp per block.
template <int QO_fragments, int d_head_accum_fragments, typename accum_t = float>
FA_DEVICE_CONSTEXPR void
final_softmax_normalization(
    accum_t (&O_accum)[QO_fragments][d_head_accum_fragments],
    accum_t (&l)[QO_fragments]) {
    // Finish summing row_sums across the 4 threads that share a row.
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
    }
    // Hoist reciprocal outside inner loop: 1 rcp + N muls  (vs N divides)
    FA_UNROLL
    for (int q = 0; q < QO_fragments; ++q) {
        const accum_t l_rcp = __frcp_rn(l[q]);
        FA_UNROLL
        for (int d = 0; d < d_head_accum_fragments; ++d)
            O_accum[q][d] *= l_rcp;
    }
}

} // namespace flash
