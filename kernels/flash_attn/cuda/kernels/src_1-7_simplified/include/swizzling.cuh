#pragma once

#include "common.cuh"

// =============================================================================
// swizzling.cuh — XOR-based smem swizzle to avoid bank conflicts
//
// The trick: instead of placing row r, col c at offset (r * cols + c),
// we XOR the column index with bits of the row index so that threads in the
// same warp access different banks.
// =============================================================================

namespace flash {

// Return swizzled column fragment index.
// Keeps the swizzled column within the same 8-element (ELEMS_PER_VEC4_ACCESS) window.
template <int col_fragments>
FA_DEVICE_CONSTEXPR int swizzled_col_fragment(int row, int col_fragment) {
    static_assert(col_fragments % ELEMS_PER_VEC4_ACCESS == 0,
                  "# col tiles must be a multiple of ELEMS_PER_VEC4_ACCESS");
    // XOR the lower bits of the row with the column fragment index.
    return (row % ELEMS_PER_VEC4_ACCESS) ^ col_fragment;
}

// Return the smem column fragment, optionally applying the swizzle.
// Use this everywhere you compute an smem address from (row, col_fragment).
template <int col_fragments, bool swizzle>
FA_DEVICE_CONSTEXPR int get_smem_col_fragment(int row, int col_fragment) {
    if constexpr (swizzle)
        return swizzled_col_fragment<col_fragments>(row, col_fragment);
    else
        return col_fragment;
}

} // namespace flash
