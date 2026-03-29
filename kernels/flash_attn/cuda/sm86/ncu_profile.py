"""Minimal script for NCU profiling — runs a single kernel call, no warmup loop."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

TARGET = sys.argv[1] if len(sys.argv) > 1 else "v1"

B, H, N, D = 1, 4, 1024, 64
q = torch.randn(B, H, N, D, device='cuda')
k = torch.randn(B, H, N, D, device='cuda')
v = torch.randn(B, H, N, D, device='cuda')

if TARGET == "v1":
    import importlib.util
    spec = importlib.util.spec_from_file_location("v1", os.path.join(os.path.dirname(__file__), "fp32_flash_attn_sm86.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    fn = mod.flash_attn
else:
    import importlib.util
    spec = importlib.util.spec_from_file_location("v2", os.path.join(os.path.dirname(__file__), "fp32_flash_attn_sm86_wmma.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    fn = mod.flash_attn_v2

# compile
fn(q, k, v); torch.cuda.synchronize()
# profile this call
fn(q, k, v); torch.cuda.synchronize()
print(f"profile target={TARGET} done")
