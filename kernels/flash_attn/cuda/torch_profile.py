"""torch.profiler kernel timing (no hardware perf counter access needed)."""
import sys, os, importlib.util
import torch
from torch.profiler import profile, record_function, ProfilerActivity

_dir = os.path.dirname(os.path.abspath(__file__))
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

v1 = _load("v1", os.path.join(_dir, "fp32_flash_attn_sm86.py"))
v2 = _load("v2", os.path.join(_dir, "fp32_flash_attn_sm86_wmma.py"))

B, H, N, D = 1, 4, 1024, 64
q = torch.randn(B, H, N, D, device='cuda')
k = torch.randn(B, H, N, D, device='cuda')
v = torch.randn(B, H, N, D, device='cuda')

# warm up jit compilation
v1.flash_attn(q, k, v); v2.flash_attn_v2(q, k, v); torch.cuda.synchronize()

for label, fn in [("v1 (baseline)", v1.flash_attn), ("v2 (float4+fp16shmem)", v2.flash_attn_v2)]:
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        with record_function(label):
            for _ in range(10): fn(q, k, v)
        torch.cuda.synchronize()
    print(f"\n=== {label} ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
