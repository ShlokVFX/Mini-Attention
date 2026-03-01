import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
import math

class GQA_Flash(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()

        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.d_model  = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads

        #linear proj

        self.w_q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2*self.head_dim*n_kv_heads)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):

        B, S, _ = x.shape

        #split heads
        q = self.w_q(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1,2) #B,H,S,dh

        kv = self.kv(x).reshape(B,S, 2, self.n_kv_heads, self.head_dim).permute(2,0,3,1,4)#2,B,H,S,dh

        k,v = kv.unbind(0)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        output = scaled_dot_product_attention(
            q,k,v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )

        output = output.transpose(1,2).contiguous().reshape(B,S,self.d_model)
        output = self.w_o(output)

        return output

if __name__ == "__main__":

    B, S, D = 2, 128, 512
    H = 8
    KV_H = 2 

    device = "cuda"

    x = torch.randn(B, S, D, device=device, dtype=torch.float16)

    gqa = GQA_Flash(
        d_model=D,
        n_heads=H,
        n_kv_heads=KV_H
    ).to(device).half()

    out = gqa(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)