import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
import math


class CombinedQKV_Flash(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model , 3*d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, S, 3, self.n_heads,self.head_dim).permute(2,0,3,1,4) #3,B,H,S,D
        q,k,v = qkv.unbind(0)

        from torch.nn.attention import sdpa_kernel,SDPBackend
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]): #can be skipped just for educational purposes , Forced FLASH ATTENTION no fallback
            output = scaled_dot_product_attention(
                q,k,v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
        
        output = output.transpose(1,2).contiguous()
        output = output.view(B,S,self.d_model)

        output = self.w_o(output)
        return output

if __name__ == "__main__":

    B, S, D = 2, 128, 512
    H = 8

    device = "cuda"

    x = torch.randn(B, S, D, device=device, dtype=torch.float16)

    mha = CombinedQKV_Flash(
        d_model=D,
        n_heads=H,
    ).to(device).half()

    out = mha(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)