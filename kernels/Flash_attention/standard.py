import torch
import torch.nn.functional as F

def standard_attention(q, k, v):
    # Get the dimension of the key vectors (last dimension of q)
    d_k = q.size(-1)
    
    # Compute scaled dot-product attention scores
    # Multiply Q with K^T and scale by sqrt(d_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=q.dtype, device=q.device)
    )
    
    # Apply softmax to obtain attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Compute weighted sum of values
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights


# Demonstrate memory bottleneck
def demonstrate_memory_issue():
    """Show the memory consumption problem of traditional Attention"""
    
    batch_size, seq_len, d_model = 2, 4096, 64
    
    # Simulated input tensors
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    # Calculate memory required for the attention matrix
    # float32 uses 4 bytes per element
    attention_matrix_size = batch_size * seq_len * seq_len * 4
    
    print(f"Sequence length: {seq_len}")
    print(f"Attention matrix size: {attention_matrix_size / (1024**2):.2f} MB")
    
    # Try running the computation to verify feasibility
    try:
        output, weights = standard_attention(q, k, v)
        print("Computation completed successfully")
    except RuntimeError as e:
        print(f"Memory error: {e}")


demonstrate_memory_issue()