# =============================
# 6. Data Statistics and Result Analysis
# =============================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib import font_manager

# ✅ Global font configuration
font_path = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
font_prop = font_manager.FontProperties(fname=font_path)
font_name = font_prop.get_name()

matplotlib.rcParams['font.family'] = font_name
matplotlib.rcParams['axes.unicode_minus'] = False

# Also configure seaborn to use the same font
sns.set(style="whitegrid")
sns.set_context("notebook", font_scale=1.0)

# -----------------------------
# Fill in the measured values from previous experiments
# -----------------------------
results = [
    {
        "Mode": "KV Cache Disabled",
        "Average Latency (s)": 0.0111,
        "Peak Memory (MB)": 430.81,
        "Total Time (s)": 0.56,
    },
    {
        "Mode": "KV Cache Enabled",
        "Average Latency (s)": 0.0074,
        "Peak Memory (MB)": 421.29,
        "Total Time (s)": 0.37,
    },
    {
        "Mode": "PagedAttention",
        "Average Latency (s)": 0.0013,
        "Peak Memory (MB)": 413.98,
        "Total Time (s)": 0.0716,
    },
]

# -----------------------------
# Build DataFrame
# -----------------------------
df = pd.DataFrame(results)
print("✅ Experiment Summary:")
print(df.to_string(index=False))
print()

# -----------------------------
# Visualization 1: Memory & Latency Comparison
# -----------------------------
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

x_pos = range(len(df))
modes = df["Mode"].tolist()

bar_colors = ["#F5B041", "#58D68D", "#5DADE2"]
ax1.bar(x_pos, df["Peak Memory (MB)"], color=bar_colors, alpha=0.7, label="Peak Memory (MB)")
ax2.plot(x_pos, df["Average Latency (s)"], color="#C0392B",
         marker="o", linewidth=2, markersize=8, label="Average Latency (s)")

ax1.set_xticks(list(x_pos))
ax1.set_xticklabels(modes, fontproperties=font_prop)

ax1.set_xlabel("Inference Optimization Mode", fontsize=11, fontproperties=font_prop)
ax1.set_ylabel("Peak Memory (MB)", fontsize=11, fontproperties=font_prop)
ax2.set_ylabel("Average Latency (s)", fontsize=11, fontproperties=font_prop)
ax1.set_title("Performance Comparison of Different Inference Strategies",
              fontsize=13, fontproperties=font_prop)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop=font_prop)

ax1.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------
# Visualization 2: Normalized Trend Comparison
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

df_normalized = df.copy()
df_normalized["Average Latency (%)"] = (
    df["Average Latency (s)"] / df["Average Latency (s)"].max() * 100
)
df_normalized["Peak Memory (%)"] = (
    df["Peak Memory (MB)"] / df["Peak Memory (MB)"].max() * 100
)

x_pos = range(len(df))
ax.plot(x_pos, df_normalized["Average Latency (%)"],
        marker="o", linewidth=2, markersize=8,
        label="Average Latency (Normalized %)", color="#E74C3C")
ax.plot(x_pos, df_normalized["Peak Memory (%)"],
        marker="s", linewidth=2, markersize=8,
        label="Peak Memory (Normalized %)", color="#5DADE2")

ax.set_xticks(list(x_pos))
ax.set_xticklabels(modes, fontproperties=font_prop)

ax.set_title("Latency and Memory Trend (Normalized)",
             fontproperties=font_prop, fontsize=13)
ax.set_xlabel("Mode", fontproperties=font_prop, fontsize=11)
ax.set_ylabel("Normalized Value (%)", fontproperties=font_prop, fontsize=11)

ax.legend(prop=font_prop)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -----------------------------
# Visualization 3: Raw Trend (Dual Y-Axis)
# -----------------------------
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

x_pos = range(len(df))
ax1.plot(x_pos, df["Average Latency (s)"],
         marker="o", linewidth=2, markersize=8,
         label="Average Latency (s)", color="#E74C3C")
ax2.plot(x_pos, df["Peak Memory (MB)"],
         marker="s", linewidth=2, markersize=8,
         label="Peak Memory (MB)", color="#5DADE2")

ax1.set_xticks(list(x_pos))
ax1.set_xticklabels(modes, fontproperties=font_prop)

ax1.set_title("Latency and Memory Trend (Raw Values)",
              fontproperties=font_prop, fontsize=13)
ax1.set_xlabel("Mode", fontproperties=font_prop, fontsize=11)
ax1.set_ylabel("Average Latency (s)", fontproperties=font_prop,
               fontsize=11, color="#E74C3C")
ax2.set_ylabel("Peak Memory (MB)", fontproperties=font_prop,
               fontsize=11, color="#5DADE2")

ax1.tick_params(axis='y', labelcolor="#E74C3C")
ax2.tick_params(axis='y', labelcolor="#5DADE2")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', prop=font_prop)

ax1.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -----------------------------
# Summary Conclusion
# -----------------------------
no_cache = df.loc[df["Mode"] == "KV Cache Disabled"].iloc[0]
with_cache = df.loc[df["Mode"] == "KV Cache Enabled"].iloc[0]
paged = df.loc[df["Mode"] == "PagedAttention"].iloc[0]

speedup_kv = no_cache["Average Latency (s)"] / with_cache["Average Latency (s)"]
speedup_paged = with_cache["Average Latency (s)"] / paged["Average Latency (s)"]

print("\n" + "="*60)
print("📊 Performance Summary")
print("="*60)
print(f"\n➡️  KV Cache vs No Cache: ~{speedup_kv:.2f}x speedup")
print(f"➡️  PagedAttention vs KV Cache: ~{speedup_paged:.2f}x speedup")
print(f"➡️  Memory usage differences are small, but latency reduction is significant")
print("\n" + "-"*60)
print("💡 Experimental Conclusion:")
print("-"*60)
print("   • KV Cache effectively reduces redundant computation and improves inference speed")
print("   • PagedAttention further improves memory and compute efficiency")
print("     and is more suitable for long-sequence or large-model inference")
print("="*60)