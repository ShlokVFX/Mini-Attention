import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


# ============================================================
# Expert Network
# ============================================================
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Standard MoE (all experts permanently on GPU)
# ============================================================
class SimpleMoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts)

        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        gate_logits = self.gate(x_flat)
        top1 = torch.argmax(gate_logits, dim=-1)

        output = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            mask = top1 == i
            if mask.any():
                output[mask] = expert(x_flat[mask])

        return output.view(B, S, -1)


# ============================================================
# Device-Aware MoE (experts live on CPU, moved only when used)
# ============================================================
class DeviceAwareMoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_experts, gpu_device):
        super().__init__()
        self.num_experts = num_experts
        self.gpu_device = gpu_device

        # Gating stays on GPU
        self.gate = nn.Linear(input_dim, num_experts).to(gpu_device)

        # Experts stay on CPU initially
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim).cpu()
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        gate_logits = self.gate(x_flat)
        top1 = torch.argmax(gate_logits, dim=-1)

        output = torch.zeros_like(x_flat)

        # Only move experts that are actually used
        active_experts = torch.unique(top1)

        for i in active_experts:
            i = i.item()
            mask = top1 == i
            if mask.any():
                expert = self.experts[i].to(self.gpu_device)
                output[mask] = expert(x_flat[mask])
                expert.cpu()  # move back immediately

        return output.view(B, S, -1)


# ============================================================
# Memory Test
# ============================================================
def test_memory_usage():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    # Larger model to show clear difference
    input_dim = 512
    output_dim = 512
    hidden_dim = 4096
    num_experts = 16
    batch_size = 2
    seq_len = 16

    dummy_input = torch.randn(batch_size, seq_len, input_dim).to(device)
    print(f"Input shape: {dummy_input.shape}")

    # ==================================================
    # Standard MoE
    # ==================================================
    print("\n" + "=" * 50)
    print("Testing Standard SimpleMoELayer (all experts on GPU)")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    standard_moe = SimpleMoELayer(
        input_dim, output_dim, hidden_dim, num_experts
    ).to(device)

    with torch.no_grad():
        out_std = standard_moe(dummy_input)

    torch.cuda.synchronize()
    peak_std = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    print(f"Peak GPU Memory: {peak_std:.2f} MB")

    del standard_moe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # ==================================================
    # Device-Aware MoE
    # ==================================================
    print("\n" + "=" * 50)
    print("Testing DeviceAwareMoELayer (experts dynamically moved)")

    device_aware_moe = DeviceAwareMoELayer(
        input_dim, output_dim, hidden_dim, num_experts, gpu_device=device
    )

    with torch.no_grad():
        out_da = device_aware_moe(dummy_input)

    torch.cuda.synchronize()
    peak_da = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    print(f"Peak GPU Memory: {peak_da:.2f} MB")

    print("\nOutput shape from standard MoE:", out_std.shape)
    print("Output shape from device-aware MoE:", out_da.shape)

    del device_aware_moe, out_std, out_da, dummy_input
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test_memory_usage()