def reference_solution(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    PyTorch implementation of Scaled Dot-Product Attention.
        
    Args:
        query: Query tensor of shape (batch, heads, seq_len, embed_dim)
        key: Key tensor of shape (batch, heads, seq_len, embed_dim)
        value: Value tensor of shape (batch, heads, seq_len, embed_dim)
            
    Returns:
        Result of scaled dot-product attention
    """
    with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=query.dtype):
        # Use PyTorch's built-in scaled dot product attention
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
    