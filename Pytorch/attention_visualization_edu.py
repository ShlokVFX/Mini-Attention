
"""
Educational Visualization Script for Scaled Dot-Product Attention
Connects to: mha_implementations_cuda.py

This script prints tensor shapes, intermediate steps,
and attention score distributions for learning purposes.
"""

import torch
import math

# Safe CUDA device setup (RTX 3060 compatible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


def reference_solution(query, key, value):
    """
    PyTorch Scaled Dot‑Product Attention (reference)
    """
    with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=query.dtype):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )


def visualize_attention(batch=1, heads=2, seq_len=4, embed_dim=8):
    print("=" * 60)
    print("Scaled Dot‑Product Attention — Educational Visualization")
    print("=" * 60)

    # Random tensors
    query = torch.randn(batch, heads, seq_len, embed_dim, device=device)
    key   = torch.randn(batch, heads, seq_len, embed_dim, device=device)
    value = torch.randn(batch, heads, seq_len, embed_dim, device=device)

    print("\nInput Tensor Shapes")
    print("-" * 30)
    print(f"Query : {query.shape}")
    print(f"Key   : {key.shape}")
    print(f"Value : {value.shape}")

    # ---- Step 1: QK^T ----
    scores = torch.matmul(query, key.transpose(-2, -1))

    print("\nRaw Attention Scores (QK^T)")
    print("-" * 30)
    print(scores[0, 0])  # print first batch, first head

    # ---- Step 2: Scaling ----
    scale = 1.0 / math.sqrt(embed_dim)
    scaled_scores = scores * scale

    print("\nScaled Scores (div √d_k)")
    print("-" * 30)
    print(scaled_scores[0, 0])

    # ---- Step 3: Softmax ----
    attn_weights = torch.softmax(scaled_scores, dim=-1)

    print("\nAttention Weights (Softmax)")
    print("-" * 30)
    print(attn_weights[0, 0])

    print("\nRow‑wise sum (should be 1.0)")
    print(attn_weights[0, 0].sum(dim=-1))

    # ---- Step 4: Weighted Sum ----
    output_manual = torch.matmul(attn_weights, value)

    print("\nManual Attention Output")
    print("-" * 30)
    print(output_manual[0, 0])

    # ---- Reference PyTorch ----
    output_ref = reference_solution(query, key, value)

    print("\nPyTorch SDPA Output")
    print("-" * 30)
    print(output_ref[0, 0])

    # ---- Difference Check ----
    diff = (output_manual - output_ref).abs().max()

    print("\nMax Difference (Manual vs Reference):", diff.item())
    print("=" * 60)


if __name__ == "__main__":
    visualize_attention()
