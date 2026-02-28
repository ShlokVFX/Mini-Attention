"""
vLLM PagedAttention Inference Script
Runs N inference steps and plots per-step memory & latency trends.
"""

import time
import subprocess
import torch
import matplotlib
matplotlib.use("Agg")  # headless — saves PNG instead of GUI popup
import matplotlib.pyplot as plt

# -----------------------------
# Monkey-patch vLLM's broken get_cached_tokenizer
# -----------------------------
import vllm.transformers_utils.tokenizer as _vllm_tok

_original_get_cached = _vllm_tok.get_cached_tokenizer

def _patched_get_cached_tokenizer(tokenizer):
    if not hasattr(tokenizer, "all_special_tokens_extended"):
        tokenizer.all_special_tokens_extended = getattr(
            tokenizer, "all_special_tokens", []
        )
    return _original_get_cached(tokenizer)

_vllm_tok.get_cached_tokenizer = _patched_get_cached_tokenizer
# -----------------------------

from vllm import LLM, SamplingParams

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Each prompt = one step on the trend charts
PROMPTS = [
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is attention in transformers?\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nExplain KV cache in one paragraph.\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is PagedAttention and how does it save GPU memory?\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nHow does vLLM achieve high throughput inference?\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nCompare eager vs chunked prefill in LLM serving.\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat are the memory bottlenecks in LLM inference?\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nDescribe Flash Attention and its benefits.\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nHow does tensor parallelism work for large models?\n<|assistant|>\n",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=50,
)

# -----------------------------
# GPU memory helper (OS-level via nvidia-smi)
# Captures vLLM's EngineCore subprocess allocations.
# torch.cuda.max_memory_allocated() only sees the main process — useless here.
# -----------------------------
def get_gpu_memory_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return sum(float(x.strip()) for x in out.strip().splitlines())
    except Exception:
        return 0.0

# -----------------------------
# Initialize model
# -----------------------------
print("🚀 Initializing vLLM model (PagedAttention enabled)...")
llm = LLM(
    model=MODEL_NAME,
    dtype="float16",
    gpu_memory_utilization=0.85,
    max_model_len=512,
)
print(f"✅ Model '{MODEL_NAME}' loaded with PagedAttention.")

print("💾 Saving tokenizer/config snapshot...")
tokenizer = llm.get_tokenizer()
tokenizer.save_pretrained("./saved_model_tokenizer")
print("✅ Tokenizer saved to ./saved_model_tokenizer\n")

# -----------------------------
# Per-step inference — collect memory + latency at each step
# -----------------------------
print("🚀 Start inference (PagedAttention mode)...")

mem_usages = []   # GPU MB at end of each step
latencies  = []   # wall-clock seconds per step
results    = []   # generated texts

for i, prompt in enumerate(PROMPTS):
    t0 = time.time()
    output = llm.generate([prompt], sampling_params)
    t1 = time.time()

    step_latency = t1 - t0
    step_mem     = get_gpu_memory_mb()   # total GPU used (model + KV cache)

    mem_usages.append(step_mem)
    latencies.append(step_latency)
    results.append(output[0].outputs[0].text.strip())

    print(f"  Step {i+1:2d}/{len(PROMPTS)} | "
          f"latency: {step_latency:.3f}s | "
          f"mem: {step_mem:.0f} MB | "
          f"snippet: {results[-1][:45]!r}...")

total_time = sum(latencies)
max_mem    = max(mem_usages)

# Main inference result (step 1 prompt)
generated_text = results[0].strip()

print("\n✅ Experiment completed (PagedAttention)")
print(f"Total runtime:      {total_time:.4f} seconds")
print(f"Peak memory usage:  {max_mem:.2f} MB")
print(f"Generated text snippet:\n{generated_text[:150]}...")

# -----------------------------
# Visualization: Memory & Latency Trend
# -----------------------------
steps = list(range(1, len(PROMPTS) + 1))

fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()

line1, = ax1.plot(steps, mem_usages, color="#4C72B0", linewidth=2.5,
                  marker="o", markersize=6, label="Memory (MB)")
line2, = ax2.plot(steps, latencies,  color="#DD8452", linewidth=2.5,
                  marker="s", markersize=6, linestyle="--", label="Latency (s)")

# Shade memory area under curve
ax1.fill_between(steps, mem_usages, alpha=0.12, color="#4C72B0")

# Annotate peak memory
peak_idx = mem_usages.index(max(mem_usages))
ax1.annotate(f"Peak\n{max(mem_usages):.0f} MB",
             xy=(steps[peak_idx], mem_usages[peak_idx]),
             xytext=(steps[peak_idx] + 0.35, mem_usages[peak_idx] - 120),
             fontsize=8, color="#4C72B0",
             arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.2))

# Annotate fastest step
fast_idx = latencies.index(min(latencies))
ax2.annotate(f"Fastest\n{min(latencies):.3f}s",
             xy=(steps[fast_idx], latencies[fast_idx]),
             xytext=(steps[fast_idx] + 0.35, latencies[fast_idx] + 0.06),
             fontsize=8, color="#DD8452",
             arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.2))

ax1.set_xlabel("Inference Step (prompt index)", fontsize=11)
ax1.set_ylabel("GPU Memory Used (MB)", color="#4C72B0", fontsize=11)
ax2.set_ylabel("Step Latency (s)",     color="#DD8452", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax2.tick_params(axis="y", labelcolor="#DD8452")
ax1.set_xticks(steps)

plt.title("Inference Performance Trend (PagedAttention / KV Cache Enabled)",
          fontsize=12, fontweight="bold", pad=12)
ax1.grid(True, linestyle="--", alpha=0.4)
fig.legend(handles=[line1, line2], loc="upper right",
           bbox_to_anchor=(0.88, 0.88), fontsize=10, framealpha=0.9)

plt.tight_layout()
out_path = "paged_attention_perf.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n📊 Chart saved → {out_path}")
plt.close()
