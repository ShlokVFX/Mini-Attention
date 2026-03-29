# =============================
# KV Cache Disabled Test (True Baseline)
# =============================
import torch
import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager

from models.load_model import load_model

model, tokenizer, device = load_model()
model.eval()

# ✅ Global font configuration
font_path = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
font_prop = font_manager.FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False

# -----------------------------
# Input text
# -----------------------------
prompt = "深度学习中的注意力机制是一种"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# -----------------------------
# Generation parameters
# -----------------------------
generate_length = 800
temperature = 0.7

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

# -----------------------------
# Step-by-step generation (NO KV Cache)
# -----------------------------
start_time = time.time()
latencies = []
mem_usages = []

output_ids = input_ids.clone()

print("🚀 Start generating (KV Cache disabled)...")

for i in range(generate_length):

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        # 🔴 FULL sequence forwarded every step
        outputs = model(
            output_ids,
            use_cache=False   # ← explicitly disabled
        )

        next_token_logits = outputs.logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    output_ids = torch.cat([output_ids, next_token], dim=1)

    torch.cuda.synchronize()
    latency = time.time() - t0

    latencies.append(latency)
    mem_usages.append(torch.cuda.max_memory_allocated() / 1024**2)

end_time = time.time()

# -----------------------------
# Result statistics
# -----------------------------
avg_latency = np.mean(latencies)
max_mem = torch.cuda.max_memory_allocated() / 1024**2
total_time = end_time - start_time

decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("✅ Experiment completed (KV Cache disabled)")
print(f"Average inference latency: {avg_latency:.4f} sec / token")
print(f"Peak memory usage: {max_mem:.2f} MB")
print(f"Total runtime: {total_time:.2f} sec")
print(f"Generated text snippet:\n{decoded[:150]}...")

# -----------------------------
# Visualization: Memory & Latency Trend
# -----------------------------
fig, ax1 = plt.subplots(figsize=(7,4))
ax2 = ax1.twinx()

ax1.plot(mem_usages, label="Memory (MB)")
ax2.plot(latencies, label="Latency (s)")

ax1.set_xlabel("Generation Step")
ax1.set_ylabel("Memory (MB)")
ax2.set_ylabel("Latency (s)")
plt.title("Inference Performance Trend (KV Cache Disabled)")
ax1.grid(True, linestyle="--", alpha=0.5)

plt.show()