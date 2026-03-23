#!/usr/bin/env python3
"""
gen_inline.py — convert flash-attention kernel source dirs into self-contained
Python files, each embedding the full inlined CUDA as a bzip2+base64 string.

Inspired by RadeonFlow/RadeonFlow_Kernels/scripts/gen_submission.py.

Usage (from repo root, WSL):
    python kernels/flash_attn/cuda/gen_inline.py            # all K1-K16
    python kernels/flash_attn/cuda/gen_inline.py --iter 16  # just K16
    python kernels/flash_attn/cuda/gen_inline.py --iter 1,7

Output: kernels/flash_attn/cuda/inline/fp16_k{N}_sm86.py

Run a generated file:
    LD_LIBRARY_PATH=... python kernels/flash_attn/cuda/inline/fp16_k16_sm86.py
"""

import argparse
import base64
import bz2
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KDIR = _HERE / "kernels"
_OUT  = _HERE / "inline"

sys.path.insert(0, str(_HERE / "py"))
from flash_helpers.kernel_configs import get_kernel_progression_configs, DType

# ── Iteration table ──────────────────────────────────────────────────────────
# (iter, src_dir, label, progression_index_or_None)
ITERS = [
    (1,  "src_1-7", "Base Implementation",          0),
    (2,  "src_1-7", "Swizzling",                    1),
    (3,  "src_1-7", "Eagerly Loading K & V",         2),
    (4,  "src_1-7", "Interleaving LD/ST",            3),
    (5,  "src_1-7", "Double Buffering",              4),
    (6,  "src_1-7", "Improving FP32 Throughput",     5),
    (7,  "src_1-7", "Auto-Tuning",                   6),
    (8,  "src_8",   "Reducing IADD3/LOP3/SHF",       None),
    (9,  "src_9",   "Reducing IMAD.MOV/MOV",          None),
    (10, "src_10",  "Removing CSRZ + Opt Softmax",   None),
    (11, "src_11",  "Encoded Swizzling RF→SMEM",      None),
    (12, "src_12",  "Misc Code Changes",             None),
    (13, "src_13",  "Iterating Backwards",           None),
    (14, "src_14",  "Cache Configuration",           None),
    (15, "src_15",  "Tiling along d_head",           None),
    (16, "src_16",  "Static GMEM Stride",            None),
]

NVCC_FLAGS = [
    "-std=c++20",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "-gencode", "arch=compute_86,code=sm_86",
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]


# ── Source inlining ──────────────────────────────────────────────────────────

def inline_source(src_file: Path, inc_dir: Path, seen: set = None) -> str:
    """Recursively inline local #include "..." directives.
    System #include <...> are kept unchanged (nvcc resolves them).
    #pragma once stripped from headers to prevent duplicate-inclusion issues.
    """
    if seen is None:
        seen = set()
    abs_path = src_file.resolve()
    if abs_path in seen:
        return ""
    seen.add(abs_path)

    text = abs_path.read_text(encoding="utf-8", errors="replace")
    out = []
    for line in text.splitlines(keepends=True):
        # Strip pragma once from header files
        if abs_path.suffix in (".cuh", ".h") and re.match(r"\s*#\s*pragma\s+once\b", line):
            continue
        # Resolve local includes
        m = re.match(r'\s*#\s*include\s+"([^"]+)"', line)
        if m:
            inc_name = m.group(1)
            candidates = [abs_path.parent / inc_name, inc_dir / inc_name]
            resolved = next((p for p in candidates if p.exists()), None)
            if resolved:
                out.append(f"// >>> {inc_name}\n")
                out.append(inline_source(resolved, inc_dir, seen))
                out.append(f"// <<< {inc_name}\n")
            else:
                out.append(line)  # not found locally — keep
        else:
            out.append(line)
    return "".join(out)


def compress_source(source: str) -> bytes:
    return bz2.compress(source.encode("utf-8"), compresslevel=9)


def to_b64_lines(data: bytes) -> str:
    """Base64-encode bytes and split into 76-char lines for a Python string literal."""
    b64 = base64.b64encode(data).decode("ascii")
    lines = [b64[i:i+76] for i in range(0, len(b64), 76)]
    return "\n".join(f'    "{line}"' for line in lines)


# ── Template helpers ─────────────────────────────────────────────────────────

def cfg_block(iter_num: int, prog_idx) -> str:
    if prog_idx is not None:
        return (
            f"# Canonical config for K{iter_num} from the progression series\n"
            f"_CFG = get_kernel_progression_configs()[{prog_idx}]"
        )
    # K8-K16: pick fastest FP16 config at runtime
    return """\
def _find_best_cfg():
    cfgs = [c for c in get_kernel_configs("all") if c.dtype == DType.FP16][:40]
    B, N, H, D = 4, 2048, 16, 128
    q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")
    k, v, o = torch.randn_like(q), torch.randn_like(q), torch.empty_like(q)
    best, best_ms = None, float("inf")
    for c in cfgs:
        try:
            for _ in range(3): _ext.forward(c, q, k, v, o, False)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(3): _ext.forward(c, q, k, v, o, False)
            e1.record(); torch.cuda.synchronize()
            ms = e0.elapsed_time(e1) / 3
            if ms < best_ms: best_ms, best = ms, c
        except Exception: pass
    return best

_CFG = _find_best_cfg()"""


def make_file(iter_num: int, label: str, src_dir: str,
              prog_idx, b64_lines: str) -> str:
    nvcc_repr = repr(NVCC_FLAGS)
    cfg_code  = cfg_block(iter_num, prog_idx)

    # Imports needed by cfg_block variants
    if prog_idx is not None:
        extra_imports = ""
    else:
        extra_imports = "from flash_helpers.kernel_configs import get_kernel_configs\n"

    lines = [
        f'#!/usr/bin/env python3',
        f'"""',
        f'FP16 Flash Attention K{iter_num}: {label}  — RTX 3060 SM86',
        f'Auto-generated by gen_inline.py from kernels/{src_dir}',
        f'',
        f'Run (WSL):',
        f'  cd /mnt/d/GITHUB/Mini-Attention',
        f'  LD_LIBRARY_PATH=/root/fa_env/lib/python3.12/site-packages/torch/lib \\',
        f'    /root/fa_env/bin/python kernels/flash_attn/cuda/inline/fp16_k{iter_num}_sm86.py',
        f'"""',
        f'',
        f'import base64, bz2, os, sys, time',
        f'from pathlib import Path',
        f'',
        f'os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")',
        f'',
        f'import torch',
        f'import torch.nn.functional as F',
        f'from torch.utils.cpp_extension import load',
        f'',
        f'sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py"))',
        f'from flash_helpers.kernel_configs import (',
        f'    DType, FlashForwardKernelConfig, get_kernel_progression_configs,',
        f')',
        extra_imports,
        f'# ── Embedded CUDA source (bzip2 + base64) ────────────────────────────────',
        f'_CUDA_B64 = (',
        b64_lines,
        f')',
        f'',
        f'# ── Extract + compile ────────────────────────────────────────────────────',
        f'_BUILD = Path(__file__).resolve().parent.parent / "build" / "inline_k{iter_num}"',
        f'_BUILD.mkdir(parents=True, exist_ok=True)',
        f'_CU = _BUILD / "flash_attention.cu"',
        f'if not _CU.exists():',
        f'    _CU.write_bytes(bz2.decompress(base64.b64decode(',
        f'        "".join(_CUDA_B64.replace("\\n", "").split()))))',
        f'',
        f'_ext = load(',
        f'    name="fp16_k{iter_num}_sm86",',
        f'    sources=[str(_CU)],',
        f'    extra_cuda_cflags={nvcc_repr},',
        f'    extra_cflags=["-O3"],',
        f'    build_directory=str(_BUILD),',
        f'    verbose=False,',
        f')',
        f'',
        f'# ── Config ───────────────────────────────────────────────────────────────',
        cfg_code,
        f'',
        f'# ── Public API ───────────────────────────────────────────────────────────',
        f'def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:',
        f'    """FP16 flash attention. q/k/v: (B, N, H, D) float16."""',
        f'    out = torch.empty_like(q)',
        f'    _ext.forward(_CFG, q.contiguous(), k.contiguous(), v.contiguous(), out, False)',
        f'    return out',
        f'',
        f'# ── Self-test ────────────────────────────────────────────────────────────',
        f'if __name__ == "__main__":',
        f'    torch.manual_seed(0)',
        f'    B, N, H, D = 8, 1024, 16, 128',
        f'    q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")',
        f'    k, v = torch.randn_like(q), torch.randn_like(q)',
        f'',
        f'    ref = F.scaled_dot_product_attention(',
        f'        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)',
        f'    ).transpose(1,2)',
        f'    out = forward(q, k, v)',
        f'    err = (out - ref).abs().max().item()',
        f'    status = "OK" if err < 0.05 else "FAIL"',
        f'    print(f"K{iter_num} [{label}]  max|out-ref|={{err:.4e}}  {{status}}")',
        f'',
        f'    # Latency',
        f'    for _ in range(10): forward(q, k, v)',
        f'    torch.cuda.synchronize()',
        f'    t0 = time.perf_counter()',
        f'    for _ in range(200): forward(q, k, v)',
        f'    torch.cuda.synchronize()',
        f'    ms = (time.perf_counter()-t0)/200*1e3',
        f'',
        f'    for _ in range(10):',
        f'        F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))',
        f'    torch.cuda.synchronize()',
        f'    t0 = time.perf_counter()',
        f'    for _ in range(200):',
        f'        F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))',
        f'    torch.cuda.synchronize()',
        f'    ref_ms = (time.perf_counter()-t0)/200*1e3',
        f'',
        f'    print(f"  latency: {{ms:.3f}} ms  SDPA: {{ref_ms:.3f}} ms  rel: {{ref_ms/ms*100:.1f}}%")',
    ]
    return "\n".join(lines) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iter", default=None,
                   help="Comma-separated iteration numbers (default: all 1-16)")
    p.add_argument("--out",  default=str(_OUT),
                   help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    iters = ITERS
    if args.iter:
        wanted = {int(x) for x in args.iter.split(",")}
        iters = [i for i in iters if i[0] in wanted]

    print(f"Generating {len(iters)} file(s) → {out_dir}\n")

    for iter_num, src_dir, label, prog_idx in iters:
        src_path = _KDIR / src_dir
        cu_file  = src_path / "flash_attention.cu"
        inc_dir  = src_path / "include"

        if not cu_file.exists():
            print(f"  K{iter_num}: SKIP — {cu_file} not found")
            continue

        print(f"  K{iter_num} ({src_dir}): inlining...", end=" ", flush=True)
        inlined    = inline_source(cu_file, inc_dir)
        compressed = compress_source(inlined)
        b64_lines  = to_b64_lines(compressed)
        print(f"{len(inlined)//1024} KB  →  {len(compressed)//1024} KB compressed")

        code = make_file(iter_num, label, src_dir, prog_idx, b64_lines)
        out_file = out_dir / f"fp16_k{iter_num}_sm86.py"
        out_file.write_text(code, encoding="utf-8")

    print(f"\nDone. Test K16:")
    print(f"  cd /mnt/d/GITHUB/Mini-Attention")
    print(f"  LD_LIBRARY_PATH=... python kernels/flash_attn/cuda/inline/fp16_k16_sm86.py")


if __name__ == "__main__":
    main()
