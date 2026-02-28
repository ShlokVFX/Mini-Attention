"""
vLLM PagedAttention Inference Script
Runs N inference steps and plots per-step memory & latency trends.
"""

import time
import subprocess
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vllm.transformers_utils.tokenizer as _vllm_tok
_original_get_cached = _vllm_tok.get_cached_tokenizer
def _patched_get_cached_tokenizer(tokenizer):
    if not hasattr(tokenizer, "all_special_tokens_extended"):
        tokenizer.all_special_tokens_extended = getattr(tokenizer, "all_special_tokens", [])
    return _original_get_cached(tokenizer)
_vllm_tok.get_cached_tokenizer = _patched_get_cached_tokenizer

from vllm import LLM, SamplingParams

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

PROMPTS = [
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is attention in transformers?\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nExplain KV cache in one paragraph.\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is PagedAttention and how does it save GPU memory?\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nHow does vLLM achieve high throughput inference?\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nCompare eager vs chunked prefill in LLM serving.\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat are the memory bottlenecks in LLM inference?\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|u ser|>\nDescribe Flash Attention and its benefits.\n<|assistant|>\n",
    "<|system|>\nYou are a helpful assistant.\n<|user|>\nHow does tensor parallelism work for large models?\n<|assistant|>\n",
]

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50)

def get_gpu_memory_mb() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        return sum(float(x.strip()) for x in out.strip().splitlines())
    except Exception:
        return 0.0

print("🚀 Initializing vLLM model (PagedAttention enabled)...")
llm = LLM(model=MODEL_NAME, dtype="float16", gpu_memory_utilization=0.85, max_model_len=512)
print(f"✅ Model '{MODEL_NAME}' loaded with PagedAttention.")

print("💾 Saving tokenizer/config snapshot...")
llm.get_tokenizer().save_pretrained("./saved_model_tokenizer")
print("✅ Tokenizer saved.\n")

print("🚀 Start inference (PagedAttention mode)...")
mem_usages, latencies, results = [], [], []

for i, prompt in enumerate(PROMPTS):
    t0 = time.time()
    output = llm.generate([prompt], sampling_params)
    t1 = time.time()
    mem_usages.append(get_gpu_memory_mb())
    latencies.append(t1 - t0)
    results.append(output[0].outputs[0].text.strip())
    print(f"  Step {i+1:2d}/{len(PROMPTS)} | latency: {latencies[-1]:.3f}s | mem: {mem_usages[-1]:.0f} MB | {results[-1][:45]!r}...")

print("\n✅ Experiment completed (PagedAttention)")
print(f"Total runtime:     {sum(latencies):.4f} seconds")
print(f"Peak memory usage: {max(mem_usages):.2f} MB")
print(f"Generated text snippet:\n{results[0][:150]}...")

steps = list(range(1, len(PROMPTS) + 1))
fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()
line1, = ax1.plot(steps, mem_usages, color="#4C72B0", linewidth=2.5, marker="o", markersize=6, label="Memory (MB)")
line2, = ax2.plot(steps, latencies,  color="#DD8452", linewidth=2.5, marker="s", markersize=6, linestyle="--", label="Latency (s)")
ax1.fill_between(steps, mem_usages, alpha=0.12, color="#4C72B0")
peak_idx = mem_usages.index(max(mem_usages))
ax1.annotate(f"Peak\n{max(mem_usages):.0f} MB",
    xy=(steps[peak_idx], mem_usages[peak_idx]),
    xytext=(steps[peak_idx]+0.35, mem_usages[peak_idx]-120),
    fontsize=8, color="#4C72B0", arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.2))
fast_idx = latencies.index(min(latencies))
ax2.annotate(f"Fastest\n{min(latencies):.3f}s",
    xy=(steps[fast_idx], latencies[fast_idx]),
    xytext=(steps[fast_idx]+0.35, latencies[fast_idx]+0.06),
    fontsize=8, color="#DD8452", arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.2))
ax1.set_xlabel("Inference Step (prompt index)", fontsize=11)
ax1.set_ylabel("GPU Memory Used (MB)", color="#4C72B0", fontsize=11)
ax2.set_ylabel("Step Latency (s)", color="#DD8452", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax2.tick_params(axis="y", labelcolor="#DD8452")
ax1.set_xticks(steps)
plt.title("Inference Performance Trend (PagedAttention / KV Cache Enabled)", fontsize=12, fontweight="bold", pad=12)
ax1.grid(True, linestyle="--", alpha=0.4)
fig.legend(handles=[line1, line2], loc="upper right", bbox_to_anchor=(0.88, 0.88), fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig("paged_attention_perf.png", dpi=150, bbox_inches="tight")
print("\n📊 Chart saved → paged_attention_perf.png")
plt.close()
