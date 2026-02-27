import time
import torch
import matplotlib.pyplot as plt
import numpy as np

from kernels.FA.block_segmentation import standard_attention, flash_attention


def measure_time_cuda(fn, warmup=5, iters=10):
    """
    Accurate CUDA timing using CUDA events.
    Falls back to time.time() on CPU.
    """
    if torch.cuda.is_available():
        # Warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()
        for _ in range(iters):
            fn()
        ender.record()

        torch.cuda.synchronize()
        elapsed_ms = starter.elapsed_time(ender)
        return elapsed_ms / 1000.0 / iters  # seconds per iteration
    else:
        # CPU fallback
        start = time.time()
        for _ in range(iters):
            fn()
        return (time.time() - start) / iters


def performance_comparison():
    seq_lengths = [256, 512, 1024, 2048, 4096]

    standard_times = []
    flash_times = []
    standard_memories = []
    flash_memories = []

    d_model = 64
    batch_size = 2
    block_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Performance experiment config: "
        f"batch_size={batch_size}, d_model={d_model}, "
        f"block_size={block_size}, device={device}"
    )

    for seq_len in seq_lengths:
        print(f"\n=== Testing sequence length: {seq_len} ===")

        torch.manual_seed(42)
        q = torch.randn(batch_size, seq_len, d_model, device=device)
        k = torch.randn(batch_size, seq_len, d_model, device=device)
        v = torch.randn(batch_size, seq_len, d_model, device=device)

        # -----------------------------
        # 1️⃣ Standard Attention
        # -----------------------------
        if seq_len <= 2048:

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            def run_standard():
                standard_attention(q, k, v)

            standard_time = measure_time_cuda(run_standard)

            if torch.cuda.is_available():
                standard_memory = (
                    torch.cuda.max_memory_allocated() / (1024 ** 2)
                )
            else:
                standard_memory = (
                    batch_size * seq_len * seq_len * 4 +
                    3 * batch_size * seq_len * d_model * 4
                ) / (1024 ** 2)

        else:
            standard_time = np.nan
            standard_memory = np.nan

        standard_times.append(standard_time)
        standard_memories.append(standard_memory)

        print(
            f"Standard Attention: "
            f"avg_time={standard_time:.6f}s, "
            f"memory={standard_memory:.2f}MB"
        )

        # -----------------------------
        # 2️⃣ Flash Attention
        # -----------------------------
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        def run_flash():
            flash_attention(q, k, v, block_size=block_size)

        flash_time = measure_time_cuda(run_flash)

        if torch.cuda.is_available():
            flash_memory = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
            )
        else:
            flash_memory = (
                3 * batch_size * block_size * d_model * 4 +
                3 * batch_size * seq_len * d_model * 4
            ) / (1024 ** 2)

        flash_times.append(flash_time)
        flash_memories.append(flash_memory)

        print(
            f"Flash Attention: "
            f"avg_time={flash_time:.6f}s, "
            f"memory={flash_memory:.2f}MB"
        )

    # -----------------------------
    # 3️⃣ Plot Results
    # -----------------------------
    plt.figure(figsize=(12, 5))

    # Time plot
    plt.subplot(1, 2, 1)
    plt.plot(
        seq_lengths,
        [t if not np.isnan(t) else 0 for t in standard_times],
        'o-',
        label='Standard Attention'
    )
    plt.plot(
        seq_lengths,
        flash_times,
        's-',
        label='Flash Attention'
    )
    plt.xlabel('Sequence Length')
    plt.ylabel('Average Time (s)')
    plt.title('Computation Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Memory plot
    plt.subplot(1, 2, 2)
    plt.plot(
        seq_lengths,
        [m if not np.isnan(m) else 0 for m in standard_memories],
        'o-',
        label='Standard Attention'
    )
    plt.plot(
        seq_lengths,
        flash_memories,
        's-',
        label='Flash Attention'
    )
    plt.xlabel('Sequence Length')
    plt.ylabel('Peak Memory (MB)')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)
    plt.show()

    return (
        seq_lengths,
        standard_times,
        flash_times,
        standard_memories,
        flash_memories
    )


if __name__ == "__main__":
    results = performance_comparison()