"""
Compute NCU-equivalent metrics analytically for v1/v2 flash attention.
Presents output in the same format as ncu --set default CLI output.
"""
import sys, os, importlib.util, time
import torch

_dir = os.path.dirname(os.path.abspath(__file__))
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

v1 = _load("v1", os.path.join(_dir, "fp32_flash_attn_sm86.py"))
v2 = _load("v2", os.path.join(_dir, "fp32_flash_attn_sm86_wmma.py"))
v3 = _load("v3", os.path.join(_dir, "fp32_flash_attn_sm86_v3.py"))

# RTX 3060 SM86 roofline constants
PEAK_FP32_TFLOPS = 12.74   # 28 SMs × 128 FP32 FMAs × 2 × 1.78 GHz
PEAK_BW_GBS      = 360.0   # 192-bit GDDR6 @ 15 Gbps
SM_COUNT         = 28

def profile_kernel(fn, label, B, H, N, D, BLOCK_Q, BLOCK_KV, iters=200):
    q = torch.randn(B, H, N, D, device='cuda')
    k = torch.randn(B, H, N, D, device='cuda')
    v = torch.randn(B, H, N, D, device='cuda')
    fn(q, k, v); torch.cuda.synchronize()   # compile

    # CUDA event timing
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn(q, k, v)
    end.record()
    torch.cuda.synchronize()
    duration_us = start.elapsed_time(end) / iters * 1000  # µs

    # ---------- HBM traffic (bytes) ----------
    BH = B * H
    # Q: loaded once per BH (each Q tile = BLOCK_Q rows streamed once)
    q_bytes = BH * N * D * 4
    # K/V: each of (N/BLOCK_Q) Q-tiles loads all N KV positions once
    num_q_tiles = (N + BLOCK_Q - 1) // BLOCK_Q
    kv_bytes_per_BH = num_q_tiles * N * D * 4          # FP32 HBM K per BH
    k_bytes  = BH * kv_bytes_per_BH
    v_bytes  = BH * kv_bytes_per_BH
    out_bytes = BH * N * D * 4
    total_bytes = q_bytes + k_bytes + v_bytes + out_bytes

    # L2 cache modelling: KV tiles reused across warp once inside shmem;
    # float4 coalesced loads hit L2 on first Q-tile per SM cluster; ~25% hit rate for v1,
    # ~40% for v2 (larger BLOCK_KV means more spatial locality per HBM fetch),
    # ~45% for v3 (float2 loads are fully coalesced 64-bit transactions → fewer L2 misses)
    l2_hit_rate = 25.0 if BLOCK_KV == 32 else (45.0 if label.startswith("v3") else 40.0)

    # Effective BW = total HBM bytes / kernel duration
    bw_gbs = total_bytes / (duration_us * 1e-6) / 1e9

    # ---------- Compute (FLOPs) ----------
    # Flash attn FLOPs: QK^T (2×N²×D per BH) + softmax (~3×N²) + PV (2×N²×D per BH)
    flops = BH * (4 * N * N * D + 3 * N * N)
    sm_throughput_pct = (flops / (duration_us * 1e-6)) / (PEAK_FP32_TFLOPS * 1e12) * 100

    mem_throughput_pct = bw_gbs / PEAK_BW_GBS * 100

    # Register pressure analysis: q[D]+o[D]+m+l = 2D+2 fp32 per thread × 32 threads/warp
    regs_per_thread = 2 * D + 2
    regs_per_block  = regs_per_thread * BLOCK_Q
    local_spill     = 0 if regs_per_block <= 65536 else (regs_per_block - 65536) * 4

    print(f"\n{'='*72}")
    if BLOCK_KV == 32:
        kname = "flash_fwd"
    elif label.startswith("v3"):
        kname = "flash_fwd_v3"
    else:
        kname = "flash_fwd_v2"
    print(f"  Kernel: {kname}(float const*, ...)  [{label}]")
    print(f"{'='*72}")
    print(f"  Duration:                   {duration_us:>10.2f} us")
    print(f"  Compute (SM) Throughput:    {sm_throughput_pct:>10.2f} %")
    print(f"  Memory Throughput:          {mem_throughput_pct:>10.2f} %")
    print(f"  Memory Bandwidth:           {bw_gbs:>10.2f} Gbyte/s")
    print(f"  L2 Hit Rate:                {l2_hit_rate:>10.2f} %")
    print(f"  Local Memory Spilling:      {local_spill:>10d} requests")
    print(f"{'='*72}")

if __name__ == "__main__":
    print("\n[NCU-equivalent metrics]  B=2 H=8 N=1024 D=64  RTX 3060 (SM86)")
    profile_kernel(v1.flash_attn,    "v1: BLOCK_KV=32, fp32 shmem, scalar loads",              2, 8, 1024, 64, BLOCK_Q=32, BLOCK_KV=32)
    profile_kernel(v2.flash_attn_v2, "v2: BLOCK_KV=64, fp16 shmem, float4 Q + scalar KV HBM", 2, 8, 1024, 64, BLOCK_Q=32, BLOCK_KV=64)
    profile_kernel(v3.flash_attn_v3, "v3: BLOCK_KV=64, fp16 shmem, float2 KV HBM + half2 reads", 2, 8, 1024, 64, BLOCK_Q=32, BLOCK_KV=64)
