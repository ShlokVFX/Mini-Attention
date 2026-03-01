import torch
import torch.nn as nn
import random


# ==============================
# 1. Attention Sink Demonstration
# ==============================

def demonstrate_attention_sinks():
    """Simple demonstration of attention sink phenomenon"""
    attention_scores = torch.tensor([5.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    softmax = nn.Softmax(dim=0)
    attention_weights = softmax(attention_scores)

    print("Raw attention scores:", attention_scores)
    print("Attention weights after softmax:", attention_weights)
    print(f"First token attention share: {attention_weights[0].item() * 100:.2f}%")
    print("-" * 50)


# ==============================
# 2. Streaming Cache
# ==============================

class SimpleStreamingCache:
    """Simplified StreamingLLM-style cache"""

    def __init__(self, sink_size=4, window_size=512):
        self.sink_size = sink_size
        self.window_size = window_size
        self.key_cache = []
        self.value_cache = []
        self.token_positions = []

    def update(self, new_keys, new_values):
        if len(new_keys) != len(new_values):
            raise ValueError("Keys and values must match in length")

        start_pos = len(self.token_positions)

        self.key_cache.extend(new_keys)
        self.value_cache.extend(new_values)
        self.token_positions.extend(
            range(start_pos, start_pos + len(new_keys))
        )

        total_keep = self.sink_size + self.window_size

        if len(self.key_cache) > total_keep:
            sink_part = list(range(min(self.sink_size, len(self.key_cache))))
            recent_start = max(len(self.key_cache) - self.window_size, self.sink_size)
            recent_part = list(range(recent_start, len(self.key_cache)))

            keep_indices = sink_part + recent_part

            self.key_cache = [self.key_cache[i] for i in keep_indices]
            self.value_cache = [self.value_cache[i] for i in keep_indices]
            self.token_positions = [self.token_positions[i] for i in keep_indices]

    def get_cache(self):
        return self.key_cache, self.value_cache


# ==============================
# 3. Position Adapter
# ==============================

class PositionAdapter:
    """Handle position remapping"""

    def __init__(self):
        self.original_to_current = {}

    def update_mapping(self, current_positions):
        self.original_to_current = {
            orig: curr for curr, orig in enumerate(current_positions)
        }

    def get_relative_positions(self):
        if not self.original_to_current:
            return []

        current_indices = list(self.original_to_current.values())
        min_index = min(current_indices)
        return [idx - min_index for idx in current_indices]


# ==============================
# 4. Simple Attention
# ==============================

def simple_attention(query, key_cache, value_cache):
    """Numeric dot-product attention"""

    if not key_cache:
        return [], torch.tensor([])

    scores = []
    for k in key_cache:
        score = sum(q * k_val for q, k_val in zip(query, k))
        scores.append(score)

    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    attention_weights = torch.softmax(scores_tensor, dim=0)

    output = [0.0] * len(value_cache[0])
    for w, v in zip(attention_weights, value_cache):
        for i in range(len(v)):
            output[i] += w.item() * v[i]

    return output, attention_weights


# ==============================
# 5. Complete Streaming Demo
# ==============================

def complete_streaming_demo():
    sink_size = 4
    window_size = 8
    token_dim = 4

    cache = SimpleStreamingCache(sink_size, window_size)
    pos_adapter = PositionAdapter()

    print("Starting StreamingLLM complete demo")
    print(f"Config: sink={sink_size}, window={window_size}, dim={token_dim}")
    print("-" * 50)

    for token_idx in range(20):
        new_key = [[random.random() for _ in range(token_dim)]]
        new_value = [[random.random() for _ in range(token_dim)]]

        cache.update(new_key, new_value)
        pos_adapter.update_mapping(cache.token_positions)

        if (token_idx + 1) % 5 == 0:
            keys, _ = cache.get_cache()
            rel_pos = pos_adapter.get_relative_positions()

            print(f"Processed tokens: {token_idx + 1}")
            print(f"Cache size: {len(keys)} (target={sink_size + window_size})")
            print("Original positions:", cache.token_positions)
            print("Relative positions:", rel_pos)
            print("-" * 30)

    print("Demo finished.")
    print("Final cache size:", len(cache.key_cache))
    print("Final original positions:", cache.token_positions)
    print("=" * 50)


# ==============================
# 6. Main
# ==============================

if __name__ == "__main__":
    demonstrate_attention_sinks()
    complete_streaming_demo()