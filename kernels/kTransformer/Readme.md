In Standard : MoE All experts reside on GPU, Memory usage scales linearly with number of experts 

In Device-Aware MoE : Experts live on CPU, Only active experts are moved to GPU, With Top-1 routing and small batch size, only a few experts activate, GPU memory scales with active experts, not total experts

Output : 

    ➜  Mini-Attention git:(main) ✗ uv run python kernels/kTransformer/base.py                                                                                                 
    Using device: cuda:0
    Input shape: torch.Size([2, 16, 512])

    ==================================================
    Testing Standard SimpleMoELayer (all experts on GPU)
    Peak GPU Memory: 265.64 MB

    ==================================================
    Testing DeviceAwareMoELayer (experts dynamically moved)
    Peak GPU Memory: 25.44 MB

    Output shape from standard MoE: torch.Size([2, 16, 512])
    Output shape from device-aware MoE: torch.Size([2, 16, 512])