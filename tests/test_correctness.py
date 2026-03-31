"""
Correctness tests — FP16 Flash Attention K1–K7 vs PyTorch SDPA reference.

Max absolute error threshold: 1e-3 (FP16 accumulation budget).
All K1–K7 expected to pass. Zero local memory spilling required.

Run (WSL, RTX 3060 or RTX 5090):
    cd /mnt/d/GITHUB/Mini-Attention
    LD_LIBRARY_PATH=/root/fa_env/lib/python3.12/site-packages/torch/lib \
      /root/fa_env/bin/python tests/test_correctness.py
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TOLERANCE = 1e-3
SEED = 42
B, H, N, D = 2, 8, 512, 128   # conservative — fast to compile on first run

INLINE_DIR = Path(__file__).resolve().parent.parent / "kernels/flash_attn/cuda/sm86/inline"


def load_kernel(k: int):
    path = INLINE_DIR / f"fp16_k{k}_sm86.py"
    spec = importlib.util.spec_from_file_location(f"fp16_k{k}_sm86", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sdpa_reference(q, k, v):
    # SDPA expects (B, H, N, D); inline kernels use (B, N, H, D)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=False,
    )
    return out.transpose(1, 2)  # back to (B, N, H, D)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping")
        sys.exit(0)

    torch.manual_seed(SEED)
    q = torch.randn(B, N, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    ref = sdpa_reference(q, k, v)

    print(f"Config: B={B} H={H} N={N} D={D}  fp16  tolerance={TOLERANCE}")
    print(f"Reference: torch.nn.functional.scaled_dot_product_attention")
    print()

    failed = []
    for k_id in range(1, 8):
        mod = load_kernel(k_id)
        out = mod.forward(q, k, v)
        err = (out.float() - ref.float()).abs().max().item()
        status = "PASS" if err < TOLERANCE else "FAIL"
        print(f"  K{k_id:<2}  max|out−ref| = {err:.2e}  [{status}]")
        if err >= TOLERANCE:
            failed.append(k_id)

    print()
    if failed:
        print(f"FAILED: K{failed}")
        sys.exit(1)
    else:
        print("All K1–K7 passed.")


if __name__ == "__main__":
    main()
