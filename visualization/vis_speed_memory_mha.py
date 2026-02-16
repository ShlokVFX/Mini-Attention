import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import os

# -----------------------------
# Config
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

batch = 1
heads = 12
d_model = 768
seq_lens = [128, 256, 512, 1024]

head_dim = d_model // heads
assert d_model % heads == 0

os.makedirs("vis", exist_ok=True)


# -----------------------------
# Utils
# -----------------------------
def tensor_bytes(t):
    return t.numel() * t.element_size()


def reset_cuda():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# -----------------------------
# Vanilla MHA (correct)
# -----------------------------
def vanilla_mha(q, k, v):
    """
    Manual Multi-Head Attention

    q,k,v: (B, H, N, D)
    """

    # QK^T
    scores = torch.matmul(q, k.transpose(-2, -1))  # (B,H,N,N)
    scores = scores / (head_dim ** 0.5)

    # Softmax
    probs = torch.softmax(scores, dim=-1)

    # Attention output
    out = torch.matmul(probs, v)  # (B,H,N,D)

    return out, scores, probs


# -----------------------------
# Benchmarks
# -----------------------------
def benchmark_vanilla(q, k, v):
    reset_cuda()
    torch.cuda.synchronize()

    start = time.time()

    out, scores, probs = vanilla_mha(q, k, v)

    torch.cuda.synchronize()
    end = time.time()

    mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    time_ms = (end - start) * 1000

    bytes_moved = (
        tensor_bytes(q)
        + tensor_bytes(k)
        + tensor_bytes(v)
        + tensor_bytes(scores)
        + tensor_bytes(probs)
        + tensor_bytes(out)
    )

    bandwidth = bytes_moved / (time_ms / 1000) / (1024 ** 3)

    return mem_mb, time_ms, bandwidth


def benchmark_sdpa(q, k, v):
    reset_cuda()
    torch.cuda.synchronize()

    start = time.time()

    out = F.scaled_dot_product_attention(q, k, v)

    torch.cuda.synchronize()
    end = time.time()

    mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    time_ms = (end - start) * 1000

    bytes_moved = (
        tensor_bytes(q)
        + tensor_bytes(k)
        + tensor_bytes(v)
        + tensor_bytes(out)
    )

    bandwidth = bytes_moved / (time_ms / 1000) / (1024 ** 3)

    return mem_mb, time_ms, bandwidth


# -----------------------------
# Run Benchmark
# -----------------------------
vanilla_mem, sdpa_mem = [], []
vanilla_time, sdpa_time = [], []
vanilla_bw, sdpa_bw = [], []

for N in seq_lens:
    print(f"\nSeq Len = {N}")

    q = torch.randn(batch, heads, N, head_dim, device=device)
    k = torch.randn(batch, heads, N, head_dim, device=device)
    v = torch.randn(batch, heads, N, head_dim, device=device)

    m1, t1, b1 = benchmark_vanilla(q, k, v)
    m2, t2, b2 = benchmark_sdpa(q, k, v)

    print(f"Vanilla → Mem {m1:.1f} MB | {t1:.2f} ms | {b1:.2f} GB/s")
    print(f"SDPA    → Mem {m2:.1f} MB | {t2:.2f} ms | {b2:.2f} GB/s")

    vanilla_mem.append(m1)
    sdpa_mem.append(m2)

    vanilla_time.append(t1)
    sdpa_time.append(t2)

    vanilla_bw.append(b1)
    sdpa_bw.append(b2)


# -----------------------------
# Plots
# -----------------------------

# Memory
plt.figure(figsize=(8, 5))
plt.plot(seq_lens, vanilla_mem, marker="o", label="Vanilla MHA")
plt.plot(seq_lens, sdpa_mem, marker="o", label="SDPA")
plt.title("Memory Usage: Vanilla MHA vs SDPA")
plt.xlabel("Sequence Length")
plt.ylabel("Memory (MB)")
plt.legend()
plt.grid(True)
plt.savefig("vis/mha_memory_compare.png", dpi=150)
#plt.show()


# Speed
plt.figure(figsize=(8, 5))
plt.plot(seq_lens, vanilla_time, marker="o", label="Vanilla MHA")
plt.plot(seq_lens, sdpa_time, marker="o", label="SDPA")
plt.title("Speed: Vanilla MHA vs SDPA")
plt.xlabel("Sequence Length")
plt.ylabel("Time (ms)")
plt.legend()
plt.grid(True)
plt.savefig("vis/mha_speed_compare.png", dpi=150)
#plt.show()


# Bandwidth
plt.figure(figsize=(8, 5))
plt.plot(seq_lens, vanilla_bw, marker="o", label="Vanilla MHA")
plt.plot(seq_lens, sdpa_bw, marker="o", label="SDPA")
plt.title("Bandwidth: Vanilla MHA vs SDPA")
plt.xlabel("Sequence Length")
plt.ylabel("GB/s")
plt.legend()
plt.grid(True)
plt.savefig("vis/mha_bandwidth_compare.png", dpi=150)
#plt.show()
