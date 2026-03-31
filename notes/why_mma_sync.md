# Why mma.sync, Not WGMMA, on RTX 5090

**One sentence:** WGMMA (`wgmma.mma_async`) requires SM90+ and Tensor Memory — neither exists on SM120 consumer Blackwell, so `mma.sync.aligned` is the only tensor-core ISA available.

---

## WGMMA requires SM90 (Hopper)

WGMMA is a Hopper-class instruction introduced with the H100. It issues matrix-multiply operations asynchronously at the *warp-group* level (4 warps acting as a unit) and accumulates into Tensor Memory (TMEM), a 512 KB/SM register file separate from shared memory. The entire FlashAttention-3 and CuTe `wgmma` abstraction stack depends on this path.

SM120 is labelled "Blackwell" but is consumer silicon (GB205) that does **not** include:

- The TMEM register file
- `tcgen05.commit` / `tcgen05.mma` PTX instructions
- The warp-group synchronisation primitives WGMMA depends on

---

## What SM120 does support

```ptx
// Ampere-vintage tensor-core instructions — present on SM80/86/89/120:
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
```

The difference on SM120 vs SM86 is raw throughput — 419 TFLOP/s vs 51 TFLOP/s — not the ISA.

---

## What this means for the kernel

`kernels/flash_attn/cuda/sm_120/` uses the same fundamental `ldmatrix + mma.sync` tiling as the SM86 kernels, with SM120-specific tuning:

| Change            | Reason                                                     |
|-------------------|------------------------------------------------------------|
| Bc=128 tiles      | SM120 has 228 KB shared memory per block vs 48 KB on SM86 |
| L2::256B hints    | 96 MB L2 makes wider cache sectors profitable              |
| `__rcp_approx_ftz_f32` | New fast reciprocal unit on GB20x silicon            |

---

## Why not Triton?

Triton 3.x generates WGMMA for SM90 targets. For SM120 it falls back to `mma.sync` but exposes no control over:

- Tile size (Bc, Br)
- Per-warp swizzle patterns (needed for zero bank-conflict `ldmatrix`)
- Async-copy double buffering (needed to hide ~70% of HBM load latency)

These are the three levers that close the gap with cuDNN at short sequences. That is why the final kernel is in CUDA C++.
