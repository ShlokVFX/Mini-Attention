"""
SageAttention CUDA kernel for SM86 -- REFERENCE IMPLEMENTATION.

Status: CORRECT (max error < 0.01), SLOW (~1000ms vs 4ms for optimized).

This is a thread-per-row design where each thread owns one Q token and
iterates all KV tiles serially. It proves the algorithm but lacks the
warp-level tiling needed for speed.

Root cause of slowness vs real SageAttention Triton:
  - Real SageAttention: BLOCK_M x BLOCK_N tile = 128x64 tokens; warp-level
    INT8 IMMA (tensor core) computes the tile in parallel across 4 warps.
  - This kernel: 1 thread per Q row, serial j-loop over BLOCK_N KV tokens,
    no tensor-core parallelism. Equivalent latency-wise to a naive CPU loop.

TODO for v2 (to close the gap to Triton SageAttention ~4ms):
  - Use mma.sync / WMMA for INT8 QK: compute BLOCK_M x BLOCK_N in one tensor
    core instruction (IMMA.8816) per warp rather than serial dp4a per thread.
  - Switch to BLOCK_M=128 with 4 warps (32x4=128 threads) and warp-tiled
    shared memory layout matching the real SageAttention Triton kernel.

Key design points (correctly implemented here):
  - smooth_k: subtract K.mean(dim=seq) OUTSIDE kernel (full-sequence mean)
  - per-block INT8 quantization: one scale per Q-tile, one per KV-tile
  - V stays FP16 (not quantized)
  - exp2 softmax trick: exp2f(x * log2e) -- 1 cycle vs 4 for expf
  - Online softmax (flash-attention style)
  - __dp4a: 4 INT8 MADs / instruction
"""

import os
import subprocess

# Target SM86 only to avoid compiling for all architectures (much faster).
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

def _setup_windows_build_env():
    """On Windows, load MSVC + CUDA build environment from vcvarsall.bat.
    Required for load_inline() to find cl.exe and CUDA_HOME.
    Safe to call multiple times (no-op if already set).
    """
    import platform
    if platform.system() != "Windows":
        return
    # Already have cl in PATH?
    try:
        subprocess.check_output(["where", "cl"], stderr=subprocess.DEVNULL)
        return  # already set up
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    vcvarsall = (r"C:\Program Files\Microsoft Visual Studio\2022\Professional"
                 r"\VC\Auxiliary\Build\vcvarsall.bat")
    if not os.path.isfile(vcvarsall):
        vcvarsall = (r"C:\Program Files\Microsoft Visual Studio\2022\Community"
                     r"\VC\Auxiliary\Build\vcvarsall.bat")
    if not os.path.isfile(vcvarsall):
        return  # can't find VS; let load_inline fail with its own message

    try:
        # Run vcvarsall.bat x64 then dump the environment
        out = subprocess.check_output(
            f'"{vcvarsall}" x64 > nul 2>&1 && set',
            shell=True, text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ[key.strip()] = val.strip()
    except Exception:
        pass  # best-effort; don't crash the kernel

_setup_windows_build_env()

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension as _cpp_ext
from torch.utils.cpp_extension import load_inline

# Patch CUDA_HOME if torch couldn't detect it at import time (common on Windows).
if _cpp_ext.CUDA_HOME is None:
    _cuda_default = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    if os.path.isdir(_cuda_default):
        _cpp_ext.CUDA_HOME = _cuda_default
        os.environ.setdefault("CUDA_HOME", _cuda_default)

_CUDA = r"""
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define BLOCK_M 64
#define BLOCK_N 64
// log2(e) for exp2 trick: exp(x) = exp2(x * LOG2E)
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ float warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  1));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}

// Grid: (ceil(N/BLOCK_M), H, B)
// Block: BLOCK_M threads -- thread `tid` owns Q[q_row = tile_m*BLOCK_M + tid]
// BLOCK_N == BLOCK_M (square tiles), each thread also loads KV row `tid` per tile.
//
// Shared memory layout:
//   float  reduce_buf[BLOCK_M]   -- scratch for cross-block reductions
//   float  K_scale[1]
//   int32  K_packed[BLOCK_N * (D/4)]  -- INT8 K, packed 4-wide
//   half   V_tile [BLOCK_N * D]
__global__ void sage_attn_fwd(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,   // pre-smoothed (smooth_k applied in Python)
    const __half* __restrict__ V,
          __half* __restrict__ Out,
    int stride_qb, int stride_qh, int stride_qn,
    int stride_kb, int stride_kh, int stride_kn,
    int stride_vb, int stride_vh, int stride_vn,
    int stride_ob, int stride_oh, int stride_on,
    int N, int D, float sm_scale, int causal)
{
    int tid    = threadIdx.x;                    // 0..BLOCK_M-1
    int b      = blockIdx.z;
    int h      = blockIdx.y;
    int tile_m = blockIdx.x;
    int q_row  = tile_m * BLOCK_M + tid;        // global Q token index

    const __half* Qbh = Q + b * stride_qb + h * stride_qh;
    const __half* Kbh = K + b * stride_kb + h * stride_kh;
    const __half* Vbh = V + b * stride_vb + h * stride_vh;
          __half* Obh = Out + b * stride_ob + h * stride_oh;

    // -- shared memory ------------------------------------------
    extern __shared__ float smem_f[];
    // layout: [BLOCK_M floats reduce_buf] [1 float K_scale]
    //         [BLOCK_N*(D/4) int32 K_packed] [BLOCK_N*D half V_tile]
    float*   reduce_buf = smem_f;                          // [BLOCK_M]
    float*   K_scale    = reduce_buf + BLOCK_M;            // [1]
    int32_t* K_packed   = (int32_t*)(K_scale + 1);        // [BLOCK_N * (D/4)]
    __half*  V_tile     = (__half*)(K_packed + BLOCK_N * (D/4)); // [BLOCK_N * D]

    // -- load Q row into registers (fp32) + compute per-thread |q|_max --
    float q_fp[128];
    float q_max = 0.f;
    if (q_row < N) {
        for (int d = 0; d < D; d++) {
            q_fp[d] = __half2float(Qbh[q_row * stride_qn + d]);
            q_max = fmaxf(q_max, fabsf(q_fp[d]));
        }
    }

    // -- Q per-block scale: reduce max over BLOCK_M threads --
    // 2 warps (BLOCK_M=64): warp-reduce each, then cross-warp via smem
    float wq = warp_max(q_max);
    if (tid % 32 == 0) reduce_buf[tid / 32] = wq;
    __syncthreads();
    float q_scale = 0.f;
    if (tid == 0) {
        float blk_max = 0.f;
        for (int w = 0; w < BLOCK_M / 32; w++) blk_max = fmaxf(blk_max, reduce_buf[w]);
        reduce_buf[0] = blk_max / 127.f + 1e-5f;   // reuse reduce_buf[0] as broadcast slot
    }
    __syncthreads();
    q_scale = reduce_buf[0];

    // -- quantize Q to INT8 packed int32 (4 elems per word) --
    int32_t q_int[32];   // D/4, max D=128 -> 32 words
    if (q_row < N) {
        for (int d4 = 0; d4 < D / 4; d4++) {
            auto qi = [&](int i) -> uint8_t {
                int v = __float2int_rn(q_fp[d4*4+i] / q_scale);
                return (uint8_t)(max(-128, min(127, v)) & 0xff);
            };
            q_int[d4] = qi(0) | (qi(1)<<8) | (qi(2)<<16) | (qi(3)<<24);
        }
    }

    // -- per-thread online softmax state --
    float m_i = -1e20f, l_i = 0.f;
    float o_fp[128];
    for (int d = 0; d < D; d++) o_fp[d] = 0.f;

    int n_tiles = (N + BLOCK_N - 1) / BLOCK_N;

    for (int kv = 0; kv < n_tiles; kv++) {
        int kv_start = kv * BLOCK_N;

        // -- cooperative load K + compute per-block scale --
        // thread `tid` loads K row `kv_start + tid`
        float k_fp[128];
        float k_rmax = 0.f;
        {
            int kv_g = kv_start + tid;
            if (kv_g < N) {
                for (int d = 0; d < D; d++) {
                    k_fp[d] = __half2float(Kbh[kv_g * stride_kn + d]);
                    k_rmax = fmaxf(k_rmax, fabsf(k_fp[d]));
                }
            }
        }

        // reduce K max across BLOCK_N threads -> K_scale in smem
        float wk = warp_max(k_rmax);
        if (tid % 32 == 0) reduce_buf[tid / 32] = wk;
        __syncthreads();
        if (tid == 0) {
            float blk_k = 0.f;
            for (int w = 0; w < BLOCK_M / 32; w++) blk_k = fmaxf(blk_k, reduce_buf[w]);
            K_scale[0] = blk_k / 127.f + 1e-5f;
        }
        __syncthreads();
        float k_scale = K_scale[0];

        // quantize K row tid -> store packed int32 to smem
        {
            int kv_g = kv_start + tid;
            for (int d4 = 0; d4 < D / 4; d4++) {
                auto ki = [&](int i) -> uint8_t {
                    if (kv_g >= N) return 0;
                    int v = __float2int_rn(k_fp[d4*4+i] / k_scale);
                    return (uint8_t)(max(-128, min(127, v)) & 0xff);
                };
                K_packed[tid * (D/4) + d4] = ki(0) | (ki(1)<<8) | (ki(2)<<16) | (ki(3)<<24);
            }

            // load V row tid into smem (FP16, not quantized)
            for (int d = 0; d < D; d++) {
                V_tile[tid * D + d] = (kv_g < N)
                    ? Vbh[kv_g * stride_vn + d]
                    : __float2half(0.f);
            }
        }
        __syncthreads();

        // -- QK dot + online softmax --
        float ks_combined = q_scale * k_scale * sm_scale;

        if (q_row < N) {
            for (int j = 0; j < BLOCK_N; j++) {
                int kv_g = kv_start + j;
                if (kv_g >= N) break;
                if (causal && kv_g > q_row) break;

                // __dp4a: 4 INT8 MADs per instruction
                int dot = 0;
                for (int d4 = 0; d4 < D / 4; d4++)
                    dot = __dp4a(q_int[d4], K_packed[j * (D/4) + d4], dot);

                float score = (float)dot * ks_combined;

                // exp2 softmax: exp(x) ? exp2(x * log2e), 1 SM86 cycle
                float m_new = fmaxf(m_i, score);
                float alpha = exp2f((m_i - m_new) * LOG2E);
                float p     = exp2f((score - m_new) * LOG2E);

                l_i = l_i * alpha + p;
                for (int d = 0; d < D; d++)
                    o_fp[d] = o_fp[d] * alpha + p * __half2float(V_tile[j * D + d]);
                m_i = m_new;
            }
        }
        __syncthreads();
    }

    // -- write output --
    if (q_row < N) {
        float inv_l = (l_i > 0.f) ? 1.f / l_i : 0.f;
        for (int d = 0; d < D; d++)
            Obh[q_row * stride_on + d] = __float2half(o_fp[d] * inv_l);
    }
}

at::Tensor sage_attn_fp16_cuda(at::Tensor Q, at::Tensor K, at::Tensor V,
                                bool causal, float sm_scale)
{
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == at::kHalf, "Q must be cuda fp16");
    int64_t B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    TORCH_CHECK(D == 64 || D == 128, "D must be 64 or 128");
    TORCH_CHECK(D % 4 == 0, "D must be divisible by 4");

    if (sm_scale <= 0.f) sm_scale = 1.f / sqrtf((float)D);
    auto Out = at::empty_like(Q);

    // smem: reduce_buf[BM f32] + K_scale[1 f32] + K_packed[BN*(D/4) i32] + V_tile[BN*D f16]
    size_t shmem = sizeof(float) * (BLOCK_M + 1)
                 + sizeof(int32_t) * BLOCK_N * (D/4)
                 + sizeof(__half)  * BLOCK_N * D;

    dim3 grid((N + BLOCK_M - 1) / BLOCK_M, (int)H, (int)B);
    dim3 block(BLOCK_M);

    sage_attn_fwd<<<grid, block, shmem>>>(
        (const __half*)Q.data_ptr(), (const __half*)K.data_ptr(),
        (const __half*)V.data_ptr(), (__half*)Out.data_ptr(),
        (int)Q.stride(0), (int)Q.stride(1), (int)Q.stride(2),
        (int)K.stride(0), (int)K.stride(1), (int)K.stride(2),
        (int)V.stride(0), (int)V.stride(1), (int)V.stride(2),
        (int)Out.stride(0), (int)Out.stride(1), (int)Out.stride(2),
        (int)N, (int)D, sm_scale, causal ? 1 : 0);

    return Out;
}
"""

_CPP = "at::Tensor sage_attn_fp16_cuda(at::Tensor Q, at::Tensor K, at::Tensor V, bool causal, float sm_scale);"

_ext = load_inline(
    name="fp16_sage_attn_sm86",
    cpp_sources=_CPP,
    cuda_sources=_CUDA,
    functions=["sage_attn_fp16_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_86", "-std=c++17"],
    verbose=False,
)


def sage_attn_fp16(q, k, v, causal=False, sm_scale=None, smooth_k=True):
    """
    SageAttention FP16 SM86 kernel.

    Args:
        q, k, v : (B, H, N, D) fp16. D in {64, 128}.
        causal  : bool -- upper-triangular causal mask.
        sm_scale: float -- defaults to 1/sqrt(D).
        smooth_k: bool -- subtract K.mean(dim=-2) before quantizing (matches real sageattn).

    Returns:
        out: (B, H, N, D) fp16
    """
    assert q.dtype == torch.float16
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if smooth_k:
        k = k - k.mean(dim=-2, keepdim=True)
    sm = float(sm_scale) if sm_scale is not None else -1.0
    return _ext.sage_attn_fp16_cuda(q, k, v, causal, sm)


# -- self-test ----------------------------------------------------
if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    device = "cuda"

    def bench(fn, warmup=20, reps=200):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps * 1e3

    print(f"Device: {torch.cuda.get_device_name(0)}  SM{torch.cuda.get_device_capability()}")
    print(f"{'Config':40s}  {'cuda':>8s}  {'sdpa':>8s}  {'triton_sage':>12s}  {'x-cuda':>8s}  maxerr")
    print("-" * 100)

    from sageattention import sageattn

    configs = [
        (1, 8,  1024, 64),
        (1, 48, 4096, 64),
        (1, 48, 8192, 64),
        (1, 24, 4096, 128),
    ]
    for B, H, N, D in configs:
        q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

        ref  = F.scaled_dot_product_attention(q, k, v)
        out  = sage_attn_fp16(q, k, v)
        err  = (out.float() - ref.float()).abs().max().item()

        ms_c = bench(lambda: sage_attn_fp16(q, k, v))
        ms_s = bench(lambda: F.scaled_dot_product_attention(q, k, v))
        ms_t = bench(lambda: sageattn(q, k.clone(), v, tensor_layout="HND"))

        print(f"  B={B} H={H} N={N:5d} D={D}  "
              f"{ms_c:6.2f}ms  {ms_s:6.2f}ms  {ms_t:10.2f}ms  "
              f"x{ms_s/ms_c:.2f}  {err:.5f}  {'OK' if err < 0.02 else 'WARN'}")
