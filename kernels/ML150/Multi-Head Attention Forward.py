import numpy as np
import math


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def multi_head_attention_forward(x, W_q, W_k, W_v, W_o, num_heads):

    batch, seq, d_model = x.shape
    d_k = d_model // num_heads

    # projections
    q = x @ W_q
    k = x @ W_k
    v = x @ W_v

    # split heads
    q = q.reshape(batch, seq, num_heads, d_k).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq, num_heads, d_k).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq, num_heads, d_k).transpose(0, 2, 1, 3)

    # sdpa
    scores = q @ k.transpose(0, 1, 3, 2)
    scores = scores / math.sqrt(d_k)

    weights = softmax(scores, axis=-1)

    output = weights @ v

    # concat heads
    output = output.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)

    output = output @ W_o

    return output.astype(np.float32)

if __name__ == "__main__":

    batch = 2
    seq = 4
    d_model = 8
    num_heads = 2

    assert d_model % num_heads == 0
    np.random.seed(42)

    x = np.random.randn(batch, seq, d_model).astype(np.float32)

    W_q = np.random.randn(d_model, d_model).astype(np.float32)
    W_k = np.random.randn(d_model, d_model).astype(np.float32)
    W_v = np.random.randn(d_model, d_model).astype(np.float32)
    W_o = np.random.randn(d_model, d_model).astype(np.float32)

    output = multi_head_attention_forward(
        x,
        W_q,
        W_k,
        W_v,
        W_o,
        num_heads
    )

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)

    print("\nOutput sample:\n", output[0, 0])

    # Shape assertion
    assert output.shape == (batch, seq, d_model)

    print("\n Test passed!")
