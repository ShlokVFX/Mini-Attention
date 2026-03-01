import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt


# ============================================================
# Scaled Dot Product Attention
# ============================================================

def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    d_k = query.size(-1)
    scaled_logits = matmul_qk / math.sqrt(d_k)

    if mask is not None:
        scaled_logits += (mask * -1e9)

    attention_weights = F.softmax(scaled_logits, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


# ============================================================
# Multi-Head Attention (MHA)
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.d_model = d_model

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attention, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)
        return output, attn_weights


# ============================================================
# Multi-Query Attention (MQA)
# ============================================================

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.d_model = d_model

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, self.depth)
        self.wv = nn.Linear(d_model, self.depth)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads_q(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads_q(self.wq(q), batch_size)
        k = self.wk(k).unsqueeze(1)
        v = self.wv(v).unsqueeze(1)

        scaled_attention, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)
        return output, attn_weights


# ============================================================
# Grouped Query Attention (GQA)
# ============================================================

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_groups == 0

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.depth = d_model // num_heads
        self.group_size = num_heads // num_groups
        self.d_model = d_model

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, num_groups * self.depth)
        self.wv = nn.Linear(d_model, num_groups * self.depth)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads_q(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def split_heads_kv(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_groups, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads_q(self.wq(q), batch_size)
        k = self.split_heads_kv(self.wk(k), batch_size)
        v = self.split_heads_kv(self.wv(v), batch_size)

        k = k.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1)
        v = v.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1)

        k = k.contiguous().view(batch_size, self.num_heads, -1, self.depth)
        v = v.contiguous().view(batch_size, self.num_heads, -1, self.depth)

        scaled_attention, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)
        return output, attn_weights


# ============================================================
# Multi-Latent Attention (MLA)
# ============================================================

class MultiLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_latents):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.d_model = d_model
        self.num_latents = num_latents

        self.wq = nn.Linear(d_model, d_model)
        self.latent_k = nn.Parameter(torch.randn(1, num_latents, d_model))
        self.latent_v = nn.Parameter(torch.randn(1, num_latents, d_model))
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k_latent = self.split_heads(self.latent_k.expand(batch_size, -1, -1), batch_size)
        v_latent = self.split_heads(self.latent_v.expand(batch_size, -1, -1), batch_size)

        scaled_attention, attn_weights = scaled_dot_product_attention(q, k_latent, v_latent, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = scaled_attention.contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)
        return output, attn_weights


# ============================================================
# Benchmark + Plot
# ============================================================

def benchmark_attention(attention_class, config, seq_len, batch_size=2, device="cuda"):
    d_model = config["d_model"]
    num_heads = config["num_heads"]

    if attention_class == GroupedQueryAttention:
        model = attention_class(d_model, num_heads, config["num_groups"]).to(device)
    elif attention_class == MultiLatentAttention:
        model = attention_class(d_model, num_heads, config["num_latents"]).to(device)
    else:
        model = attention_class(d_model, num_heads).to(device)

    model.eval()
    x = torch.randn(batch_size, seq_len, d_model).to(device)

    with torch.no_grad():
        _ = model(x, x, x)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(30):
            _ = model(x, x, x)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    avg_time = (end - start) / 30
    return avg_time


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    seq_len = 1024
    config = {
        "d_model": 512,
        "num_heads": 8,
        "num_groups": 2,
        "num_latents": 64,
    }

    attention_classes = [
        MultiHeadAttention,
        GroupedQueryAttention,
        MultiQueryAttention,
        MultiLatentAttention,
    ]

    times = []
    names = []

    print("\nBenchmarking...\n")

    for cls in attention_classes:
        t = benchmark_attention(cls, config, seq_len, device=device)
        times.append(t * 1000)
        names.append(cls.__name__)
        print(f"{cls.__name__:>25}: {t*1000:.2f} ms")

    # ========================================================
    # Single Plot
    # ========================================================

    plt.figure()
    plt.bar(names, times)

    plt.xlabel("Attention Type")
    plt.ylabel("Average Forward Time (ms)")
    plt.title("Attention Mechanism Trade-offs\n"
              "MHA: Baseline | GQA: Balanced | MQA: Fastest | MLA: Long-Context Efficient")

    plt.show()