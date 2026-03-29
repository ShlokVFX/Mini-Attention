# CLAUDE.md — B200 / SM_100a tcgen05 kernel (PR #11)
# Place this file at: kernels/flash_attn/cuda/b200/CLAUDE.md
# Claude Code will auto-read it at session start.

## What this directory is
Hand-written FlashAttention forward kernel for NVIDIA B200 (SM_100a / compute_100a).
Part of github.com/ShlokVFX/Mini-Attention, branch: wip-b200-kernel, PR #11.
Do NOT merge to main until forward_kernel_tcgen05.cuh is validated.

---

## Architecture constraints — read before touching any kernel

| Fact | Detail |
|------|--------|
| Target GPU | B200, SM_100a (compute capability 10.0) |
| Build flag | `-gencode arch=compute_100a,code=sm_100a` (already set in _bench_b200.py) |
| MMA instruction | **tcgen05** (Tensor Memory, TMEM-backed UMMA) |
| wgmma status | ❌ SM_90a only — forward_kernel_wgmma.cuh is a broken stub, ignore it |
| SMEM descriptor | swizzle=0 ONLY — `3ULL<<62` SWIZZLE_32B causes silent wrong results on B200 |
| TMEM launch | tcgen05.mma is issued by **one thread per CTA**, not the whole warpgroup |
| TMEM barrier | Always call `tcgen05.wait::st` before any `tcgen05.ld` |
| Block size | 128 threads (1 warpgroup = 4 warps), 1 block/SM for TMEM budget |

---

## File map — what exists, what's new

```
kernels/flash_attn/cuda/b200/
│
├── CLAUDE.md                        ← you are here
│
├── ptx_functions.cuh                ← MODIFIED THIS SESSION
│   │                                   Append the 5 tcgen05 functions below.
│   │                                   Do not remove existing mma.sync helpers.
│   └── [append]
│       tcgen05_alloc()              PTX: tcgen05.alloc.cta_group::1.sync.aligned
│       tcgen05_mma_f16()            PTX: tcgen05.mma.cta_group::1.kind::f16
│       tcgen05_wait_st()            PTX: tcgen05.wait::st.sync.aligned
│       tcgen05_ld_s()               PTX: tcgen05.ld.sync.aligned.16x64b.x32  (16 regs)
│       tcgen05_dealloc()            PTX: tcgen05.dealloc.cta_group::1.sync.aligned
│       make_smem_desc()             SmemDescriptor builder, swizzle param always 0
│
├── forward_kernel.cuh               ← DO NOT TOUCH (working mma.sync baseline)
│
├── forward_kernel_wgmma.cuh         ← DO NOT TOUCH (broken stub, SM_90a only)
│
├── forward_kernel_tcgen05.cuh       ← CREATE THIS FILE (main task)
│   Pipeline (in order):
│   1.  tcgen05_alloc  → grab TMEM cols for QK accumulator
│   2.  Load Q tile    → SMEM  (cp.async, 128B aligned, swizzle=0)
│   3.  KV loop:
│       a. Load K tile → SMEM
│       b. tcgen05_mma_f16  (QK = Q * K^T, TMEM accum, thread 0 only)
│       c. tcgen05_wait_st
│       d. tcgen05_ld_s     → 16 uint32 regs (16x64b.x32 layout)
│       e. pair-wise online softmax in registers (__half2, fp32 m/l state)
│       f. write rescaled P → SMEM  (smem_p, fp16)
│       g. Load V tile → SMEM (transposed into smem_v for column-major mma)
│       h. mma.sync m16n8k16  (O += P * V, register accumulator)
│   4.  Epilogue: divide O by row_l, write O + lse to global mem
│   5.  tcgen05_dealloc
│
│   Tile sizes:  BLK_M=64, BLK_N=64, HEAD_D=64
│   SMEM layout: smem_q + smem_k + smem_v + smem_p  ≈ 32 KB total
│   TMEM cols:   BLK_M × (HEAD_D/2) = 2048
│
├── flash_kernels.cuh                ← MODIFY THIS FILE
│   Remove:  wgmma_forward_kernels map
│   Add:     tcgen05_forward_kernels map keyed on {sm_arch=100, head_dim, is_causal}
│   Update:  dispatch_forward_kernel() → route sm_arch>=100 to tcgen05 path
│
├── _bench_b200.py                   ← DO NOT TOUCH (build flags already correct)
│
└── kernel_configs.py                ← DO NOT TOUCH unless adding new head_dim configs
```

---

## Softmax register layout (critical detail)

`tcgen05_ld_s` uses the `16x64b.x32` variant.
Each thread receives 16 × uint32 registers = 32 fp16 values.
Reinterpret as `__half2[16]` for pair-wise arithmetic.
Row-max reduction: `fmaxf` over all 16 `hmax2_to_float()` values, then warp shuffle.
Online update: `alpha = exp(old_m - new_m)`, rescale `o_accum` by alpha each KV block.

---

## Known TODOs (do not fix in this session unless asked)

- [ ] V-tile transposition in step 3g is scalar — replace with ldmatrix-based swizzled copy
- [ ] Causal mask not applied in softmax loop (tcgen05 causal dispatch is stubbed)
- [ ] Only HEAD_D=64 tuned — extend tile table for 128 and 256
- [ ] No pipelining between KV blocks (cp.async stages=1)

---

## How to build and test

```bash
# From repo root, on the B200 machine:
cd kernels/flash_attn/cuda/b200
python _bench_b200.py

# Compile check only (no Python):
nvcc -std=c++17 -arch=sm_100a \
     -I. forward_kernel_tcgen05.cuh \
     --cubin -o /dev/null
```

Expected: clean compile, no ptxas warnings about TMEM budget.
If ptxas warns "TMEM columns exceed budget": reduce BLK_M or HEAD_D in the TMEM_COLS constant.

---

## Reference files from last session (already written, may need integration)

The following files were generated in the previous Claude.ai session and should be
on disk alongside this CLAUDE.md. If they are missing, re-generate from PR #11 context:

- `ptx_functions_tcgen05_additions.cuh`  → contents to append into ptx_functions.cuh
- `forward_kernel_tcgen05.cuh`           → the new kernel file (may need edits)
- `flash_kernels.cuh`                    → updated dispatch map

If those files ARE on disk, the session task is:
1. Read ptx_functions.cuh (existing)
2. Append tcgen05 functions from ptx_functions_tcgen05_additions.cuh
3. Review forward_kernel_tcgen05.cuh — fix any compile errors
4. Replace flash_kernels.cuh with the updated version
5. Run build check above

---

## How to continue if starting fresh (files NOT on disk)

Tell Claude Code:
"Read this CLAUDE.md. Then write the tcgen05 additions to ptx_functions.cuh,
create forward_kernel_tcgen05.cuh, and update flash_kernels.cuh.
Follow the pipeline order and tile sizes in CLAUDE.md exactly."

---

## Session handoff checklist

After each session, update the checkbox state below before closing:

- [x] ptx_functions.cuh has tcgen05_alloc/mma_f16/wait_st/ld_s/dealloc + make_smem_desc
- [x] forward_kernel_tcgen05.cuh written (needs compile check on B200)
- [x] flash_kernels.cuh has tcgen05_forward_kernels map + Tcgen05KernelConfig key
- [x] flash_attention.cu exposes forward_tcgen05() via pybind11
- [ ] forward_kernel_tcgen05.cuh compiles cleanly (nvcc -arch=sm_100a)
- [ ] _bench_b200.py runs without crash
- [ ] NCU profile captured (tflops/s, SMEM utilization, TMEM column usage)
- [ ] PR #11 description updated with benchmark numbers
