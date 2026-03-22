import torch
import triton
import triton.language as tl

# Device-aware MoE: expert weights live on CPU; only active experts migrate to GPU
# GPU memory: gate_proj + down_proj for 1 active expert = 2 × D × D_ff × 2B FP16
#   LLaMA-style (D=4096, D_ff=11008): 2 × 4096 × 11008 × 2B = 180MB per expert on GPU at once
#   vs all-on-GPU with 16 experts: 16 × 180MB = 2.88GB — exceeds 3060's 12GB feasibility for large models
# Triton kernel: fused gate_proj→SiLU→down_proj for tokens routed to one expert
# SM86: BLOCK_T=64 tokens, BLOCK_D=64 head-dim; tl.dot → HMMA.16816 at 128 FP16 FMAs/warp/cycle

@triton.jit
def _expert_ffn_kernel(
    X_ptr,    # (num_tokens, D) — tokens routed to this expert
    G_ptr,    # (D_ff, D)       — gate projection weight
    U_ptr,    # (D_ff, D)       — up projection weight (SwiGLU: gate * silu(up))
    D_ptr,    # (D, D_ff)       — down projection weight
    Out_ptr,
    num_tokens, D, D_ff: tl.constexpr,
    BLOCK_T: tl.constexpr,   # token tile
    BLOCK_D: tl.constexpr,   # input/output dim tile
    BLOCK_F: tl.constexpr,   # D_ff tile
):
    pid_t = tl.program_id(0)   # token tile
    pid_d = tl.program_id(1)   # output dim tile

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # HMMA.16816: BLOCK_T×BLOCK_F tile costs BLOCK_T×D_ff×D FP16 FMAs
    # at BLOCK_T=64, D=4096, D_ff=11008: 64×11008×4096 = 2.88 GFLOPS per block (chunked)
    # out_acc initialized outside the D_ff loop — Triton can't use runtime conditionals to delay init
    out_acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    for f_start in range(0, D_ff, BLOCK_F):
        f_offs = f_start + tl.arange(0, BLOCK_F)

        gate_acc = tl.zeros([BLOCK_T, BLOCK_F], dtype=tl.float32)
        up_acc   = tl.zeros([BLOCK_T, BLOCK_F], dtype=tl.float32)

        for k_start in range(0, D, BLOCK_D):
            k_offs = k_start + tl.arange(0, BLOCK_D)

            x = tl.load(X_ptr + t_offs[:, None] * D + k_offs[None, :],
                        mask=(t_offs[:, None] < num_tokens) & (k_offs[None, :] < D), other=0.0)
            g_w = tl.load(G_ptr + f_offs[:, None] * D + k_offs[None, :],
                          mask=(f_offs[:, None] < D_ff) & (k_offs[None, :] < D), other=0.0)
            u_w = tl.load(U_ptr + f_offs[:, None] * D + k_offs[None, :],
                          mask=(f_offs[:, None] < D_ff) & (k_offs[None, :] < D), other=0.0)

            gate_acc = tl.dot(x.to(tl.float16), tl.trans(g_w).to(tl.float16), gate_acc)
            up_acc   = tl.dot(x.to(tl.float16), tl.trans(u_w).to(tl.float16), up_acc)

        # SwiGLU activation: silu(gate) * up — in registers, no HBM write
        gate_f    = gate_acc.to(tl.float32)
        up_f      = up_acc.to(tl.float32)
        silu_gate = gate_f * (1.0 / (1.0 + tl.exp(-gate_f)))
        hidden    = silu_gate * up_f   # (BLOCK_T, BLOCK_F) fp32 activated

        # down projection: accumulate hidden @ D_w[f_start:f_start+BLOCK_F, d_tile] → out_acc
        d_w = tl.load(D_ptr + f_offs[:, None] * D + d_offs[None, :],
                      mask=(f_offs[:, None] < D_ff) & (d_offs[None, :] < D), other=0.0)
        out_acc = tl.dot(hidden.to(tl.float16), d_w.to(tl.float16), out_acc)

    tl.store(Out_ptr + t_offs[:, None] * D + d_offs[None, :],
             out_acc.to(tl.float16),
             mask=(t_offs[:, None] < num_tokens) & (d_offs[None, :] < D))


def expert_ffn(x, gate_w, up_w, down_w):
    """Fused SwiGLU expert FFN for tokens routed to one expert.

    x:      (num_tokens, D) fp16
    gate_w: (D_ff, D) fp16
    up_w:   (D_ff, D) fp16
    down_w: (D, D_ff) fp16
    returns (num_tokens, D) fp16
    """
    num_tokens, D = x.shape
    D_ff = gate_w.shape[0]
    assert x.dtype == torch.float16

    out = torch.empty(num_tokens, D, device=x.device, dtype=torch.float16)

    BLOCK_T, BLOCK_D, BLOCK_F = 64, 64, 64

    grid = (triton.cdiv(num_tokens, BLOCK_T), triton.cdiv(D, BLOCK_D))
    _expert_ffn_kernel[grid](
        x, gate_w, up_w, down_w, out,
        num_tokens, D, D_ff,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D, BLOCK_F=BLOCK_F,
        num_warps=4,
    )
    return out


def device_aware_moe_forward(x, gate_proj, experts_gate, experts_up, experts_down, top_k=2):
    """Device-aware MoE: gate on GPU, expert weights streamed from CPU.

    x:             (B*T, D) input tokens
    gate_proj:     (num_experts, D) router weights (on GPU)
    experts_gate:  list of (D_ff, D) tensors (on CPU)
    experts_up:    list of (D_ff, D) tensors (on CPU)
    experts_down:  list of (D, D_ff) tensors (on CPU)
    """
    num_experts = len(experts_gate)
    num_tokens, D = x.shape

    # router: (num_tokens, num_experts) — runs entirely on GPU, O(num_tokens × num_experts × D)
    logits = torch.matmul(x, gate_proj.T)
    weights, indices = torch.topk(logits.softmax(dim=-1), top_k, dim=-1)

    out = torch.zeros_like(x)

    for expert_id in range(num_experts):
        # find tokens routed to this expert
        token_mask = (indices == expert_id).any(dim=-1)
        if not token_mask.any():
            continue

        routed_tokens = x[token_mask]

        # move expert weights CPU → GPU only when needed
        # GPU memory held: 3 × D_ff × D × 2B FP16 (one expert at a time)
        g_w = experts_gate[expert_id].to(x.device)
        u_w = experts_up[expert_id].to(x.device)
        d_w = experts_down[expert_id].to(x.device)

        expert_out = expert_ffn(routed_tokens, g_w, u_w, d_w)

        # weight by the router probability assigned to this expert for each token
        expert_idx_in_topk = (indices[token_mask] == expert_id).float().argmax(dim=-1)
        expert_weights = weights[token_mask].gather(1, expert_idx_in_topk.unsqueeze(1))
        out[token_mask] += expert_out * expert_weights

        # move weights back to CPU immediately — GPU holds only ~180MB peak per expert
        del g_w, u_w, d_w
        torch.cuda.empty_cache()

    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    num_experts = 8
    D, D_ff = 256, 512    # small for quick test; real LLaMA uses D=4096, D_ff=11008
    num_tokens = 64

    x = torch.randn(num_tokens, D, device='cuda', dtype=torch.float16)
    gate_proj    = torch.randn(num_experts, D, device='cuda', dtype=torch.float16)
    experts_gate = [torch.randn(D_ff, D, dtype=torch.float16) for _ in range(num_experts)]
    experts_up   = [torch.randn(D_ff, D, dtype=torch.float16) for _ in range(num_experts)]
    experts_down = [torch.randn(D, D_ff, dtype=torch.float16) for _ in range(num_experts)]

    out = device_aware_moe_forward(x, gate_proj, experts_gate, experts_up, experts_down)
    print(f"fp16_moe_sm86  output shape={out.shape}  mean|out|={out.abs().mean().item():.4f}")

    gpu_mb = torch.cuda.max_memory_allocated() / 1e6
    all_gpu_mb = num_experts * 3 * D_ff * D * 2 / 1e6
    print(f"  peak GPU: {gpu_mb:.1f}MB  vs all-experts-on-GPU: {all_gpu_mb:.1f}MB")
