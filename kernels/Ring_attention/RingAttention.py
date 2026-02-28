import torch
import torch.nn as nn
import numpy as np

class StandardAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(StandardAttention, self).__init__()
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
        self.d_k = d_k
        
    def forward(self, x):
        Q = self.W_q(x)  # (batch_size, seq_len, d_k)
        K = self.W_k(x)  # (batch_size, seq_len, d_k)
        V = self.W_v(x)  # (batch_size, seq_len, d_v)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(attn_weights, V)
        return output
        
# 2. Ring Attention Block Implementation
class RingAttentionBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, num_blocks):
        super(RingAttentionBlock, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_blocks = num_blocks
        
        # Initialize linear projections for Query, Key, and Value
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
        
    def forward(self, x, block_idx):
        """Forward pass of Ring Attention"""
        batch_size, block_len, _ = x.shape
        
        # Compute Query, Key, and Value for the current block
        Q = self.W_q(x)  # (batch_size, block_len, d_k)
        K = self.W_k(x)  # (batch_size, block_len, d_k)
        V = self.W_v(x)  # (batch_size, block_len, d_v)
        
        # Initialize global attention output
        output = torch.zeros((batch_size, block_len, self.d_v), device=x.device)
        
        # Simulate ring communication process (simplified: real version requires multi-device communication)
        for step in range(self.num_blocks):
            # Simplification: assume KV from all blocks are obtained via ring passing
            K_remote = K
            V_remote = V
            
            # Compute attention scores
            scores = torch.matmul(Q, K_remote.transpose(-2, -1)) / np.sqrt(self.d_k)
            attn_weights = torch.softmax(scores, dim=-1)
            
            # Accumulate attention output
            output += torch.matmul(attn_weights, V_remote)
        
        return output, (K, V)

        # 3. Distributed Ring Attention Class
class DistributedRingAttention:
    def __init__(self, d_model, d_k, d_v, num_devices):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_devices = num_devices
        
        # Initialize linear projection weights
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
    
    def process_block(self, x_block, device_idx, received_kv=None):
        """Process computation for a single block"""
        # Compute Query, Key, and Value for the current block
        Q = self.W_q(x_block)
        K = self.W_k(x_block)
        V = self.W_v(x_block)
        
        # Initialize attention output
        output = torch.zeros(
            (x_block.shape[0], x_block.shape[1], self.d_v),
            device=x_block.device
        )
        
        # Process received external KV (KV from previous device via ring passing)
        if received_kv is not None:
            K_prev, V_prev = received_kv
            scores = torch.matmul(Q, K_prev.transpose(-2, -1)) / np.sqrt(self.d_k)
            attn_weights = torch.softmax(scores, dim=-1)
            output += torch.matmul(attn_weights, V_prev)
        
        # Process local KV
        scores_local = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights_local = torch.softmax(scores_local, dim=-1)
        output += torch.matmul(attn_weights_local, V)
        
        # Prepare KV to send to the next device
        kv_to_send = (K, V)
        return output, kv_to_send

# 4. Simulated Implementation of Distributed Ring Attention
def simulate_distributed_attention(self, x):
    """Simulate distributed Ring Attention processing"""
    batch_size, seq_len, _ = x.shape
    block_len = seq_len // self.num_devices
    
    # Split input sequence into multiple blocks (corresponding to multiple devices)
    x_blocks = torch.chunk(x, self.num_devices, dim=1)
    
    # Initialize device outputs and KV cache
    device_outputs = [None] * self.num_devices
    device_kv_pairs = [None] * self.num_devices
    
    # Simulate multiple rounds of ring communication
    for round in range(self.num_devices):
        for device_idx in range(self.num_devices):
            # Determine KV source for this round (ring left shift)
            kv_source_idx = (device_idx - round) % self.num_devices
            
            # First round processes only local KV; later rounds process passed KV
            received_kv = device_kv_pairs[kv_source_idx] if round > 0 else None
            
            # Process current device's block
            output, kv_to_send = self.process_block(
                x_blocks[device_idx], device_idx, received_kv
            )
            
            # Accumulate outputs (attention results from multiple KV rounds)
            if device_outputs[device_idx] is None:
                device_outputs[device_idx] = output
            else:
                device_outputs[device_idx] += output
            
            # Cache current device's KV for next round passing
            device_kv_pairs[device_idx] = kv_to_send
    
    # Merge outputs from all devices to restore full sequence
    return torch.cat(device_outputs, dim=1)


# Attach the simulation method to the class
DistributedRingAttention.simulate_distributed_attention = simulate_distributed_attention

# 5. Performance Testing Function
def test_attention_performance():
    """Test performance of different attention mechanisms"""
    d_model, d_k, d_v = 512, 64, 64
    batch_size = 2
    num_devices = 4  # Simulate a distributed environment with 4 devices
    
    # Test sequence lengths (must be divisible by num_devices)
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    standard_times = []
    ring_times = []
    standard_memory = []
    ring_memory = []
    
    for seq_len in seq_lengths:
        print(f"\n=== Testing sequence length: {seq_len} ===")
        
        # Generate test data
        x = torch.randn(batch_size, seq_len, d_model)
        
        # ---------------------- Test Standard Attention ----------------------
        standard_attention = StandardAttention(d_model, d_k, d_v)
        
        # Measure time (disable gradient computation for speed)
        start_time = time.time()
        with torch.no_grad():
            output_std = standard_attention(x)
        end_time = time.time()
        std_time = end_time - start_time
        standard_times.append(std_time)
        
        # Memory estimation (including batch_size, float32=4 bytes, in MB)
        std_memory = (batch_size * seq_len * seq_len * 4) / (1024 ** 2)
        standard_memory.append(std_memory)
        
        print(f"Standard Attention - Time: {std_time:.4f}s, Estimated Memory: {std_memory:.2f}MB")
        
        # ---------------------- Test Ring Attention ----------------------
        ring_attention = DistributedRingAttention(d_model, d_k, d_v, num_devices)
        
        # Measure time (disable gradient computation)
        start_time = time.time()
        with torch.no_grad():
            output_ring = ring_attention.simulate_distributed_attention(x)
        end_time = time.time()
        ring_time = end_time - start_time
        ring_times.append(ring_time)
        
        # Ring Attention memory estimation (per device, block size = seq_len // num_devices)
        block_len = seq_len // num_devices
        ring_mem_per_device = (batch_size * block_len * block_len * 4) / (1024 ** 2)
        ring_memory.append(ring_mem_per_device)
        
        print(f"Ring Attention - Time: {ring_time:.4f}s, Estimated Memory per Device: {ring_mem_per_device:.2f}MB")
    
    return seq_lengths, standard_times, ring_times, standard_memory, ring_memory


import time
import matplotlib.pyplot as plt

# Run performance test
seq_lengths, std_times, ring_times, std_memory, ring_memory = test_attention_performance()

# Plot results
plt.figure(figsize=(12, 5))

# Subplot 1: Computation time comparison
plt.subplot(1, 2, 1)
plt.plot(seq_lengths, std_times, 'o-', linewidth=2, markersize=6, label='Standard Attention')
plt.plot(seq_lengths, ring_times, 's-', linewidth=2, markersize=6, label='Ring Attention (4-device simulation)')
plt.xlabel('Sequence Length', fontsize=11)
plt.ylabel('Computation Time (seconds)', fontsize=11)
plt.title('Computation Time Comparison of Different Attention Mechanisms', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(seq_lengths)

# Subplot 2: Memory usage comparison
plt.subplot(1, 2, 2)
plt.plot(seq_lengths, std_memory, 'o-', linewidth=2, markersize=6, label='Standard Attention')
plt.plot(seq_lengths, ring_memory, 's-', linewidth=2, markersize=6, label='Ring Attention (per device)')
plt.xlabel('Sequence Length', fontsize=11)
plt.ylabel('Memory Usage (MB)', fontsize=11)
plt.title('Memory Usage Comparison of Different Attention Mechanisms', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(seq_lengths)

plt.tight_layout()
plt.show()