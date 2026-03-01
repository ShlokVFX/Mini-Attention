import torch
import torch.nn.functional as F


def standard_attention(q, k, v):
    """
    Standard scaled dot-product attention.
    Computes the full attention matrix explicitly.
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=q.dtype, device=q.device)
    )
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


def flash_attention(q, k, v, block_size=512):
    """
    Memory-efficient (Flash-style) attention using block-wise processing
    and online softmax normalization to avoid materializing the full
    attention matrix.
    """
    batch_size, seq_len, d_model = q.shape
    d_k = q.size(-1)
    device = q.device
    dtype = q.dtype

    # Initialize output tensor (same type and device as input)
    output = torch.zeros((batch_size, seq_len, d_model),
                         device=device, dtype=dtype)

    # Number of blocks (ceiling division)
    num_blocks = (seq_len + block_size - 1) // block_size

    # Outer loop: iterate over Query blocks
    for i in range(num_blocks):
        start_i = i * block_size
        end_i = min((i + 1) * block_size, seq_len)

        q_block = q[:, start_i:end_i, :]  # (B, block, D)

        # Temporary variables for this Q block
        block_output = torch.zeros_like(q_block)
        block_max = torch.full(
            (batch_size, end_i - start_i, 1),
            -float('inf'),
            device=device,
            dtype=dtype
        )
        block_sum = torch.zeros(
            (batch_size, end_i - start_i, 1),
            device=device,
            dtype=dtype
        )

        # Inner loop: iterate over K/V blocks
        for j in range(num_blocks):
            start_j = j * block_size
            end_j = min((j + 1) * block_size, seq_len)

            k_block = k[:, start_j:end_j, :]
            v_block = v[:, start_j:end_j, :]

            # 1. Compute scaled attention scores for this block
            scores = torch.matmul(
                q_block, k_block.transpose(-2, -1)
            ) / torch.sqrt(torch.tensor(d_k, dtype=dtype, device=device))

            # 2. Online update of max and normalization factor
            block_max_new = torch.maximum(
                block_max,
                scores.max(dim=-1, keepdim=True).values
            )

            # Adjust previous sums based on new max
            exp_adj = torch.exp(block_max - block_max_new)

            block_sum = block_sum * exp_adj

            # Exponentiate current scores (numerically stable)
            exp_scores = torch.exp(scores - block_max_new)

            # Accumulate normalization sum
            block_sum = block_sum + exp_scores.sum(dim=-1, keepdim=True)

            # 3. Accumulate weighted value contributions
            block_output = block_output * exp_adj + torch.matmul(
                exp_scores, v_block
            )

            # Update running max
            block_max = block_max_new

        # 4. Final normalization for this Q block
        block_output = block_output / block_sum

        output[:, start_i:end_i, :] = block_output

    return output


def test_flash_attention():
    batch_size, seq_len, d_model = 2, 1024, 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(batch_size, seq_len, d_model, device=device)
    k = torch.randn(batch_size, seq_len, d_model, device=device)
    v = torch.randn(batch_size, seq_len, d_model, device=device)

    print(
        f"Test config: batch_size={batch_size}, "
        f"seq_len={seq_len}, d_model={d_model}, device={device}"
    )

    # 1. Measure standard attention memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    standard_out, _ = standard_attention(q, k, v)

    standard_mem = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else 0.0
    )

    print(f"Standard Attention peak memory: {standard_mem:.2f} MB")

    # 2. Measure Flash-style attention memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    flash_out = flash_attention(q, k, v, block_size=256)

    flash_mem = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else 0.0
    )

    print(f"Flash Attention peak memory: {flash_mem:.2f} MB")

    if standard_mem > 0:
        print(
            f"Memory reduction: "
            f"{(standard_mem - flash_mem) / standard_mem * 100:.1f}%"
        )

    # 3. Verify correctness
    diff = torch.max(torch.abs(standard_out - flash_out)).item()
    print(f"Maximum output difference: {diff:.6f}")


if __name__ == "__main__":
    test_flash_attention()