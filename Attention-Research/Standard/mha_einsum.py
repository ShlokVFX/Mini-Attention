import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CombinedQKV_Einsum(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(d_model , 3*d_model)
        self.w_o = nn.Linear(d_model , d_model)

    def forward(self, x, mask=None):

        B, S, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B,S,3,self.n_heads,self.head_dim).permute(2,0,3,1,4)
        #3,B,H,S,D
        q,k,v = qkv.unbind(0) 

        scores = torch.einsum("bhsd,bhtd->bhst",q,k) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1) #B,1,S,S
            scores = scores.masked_fill(mask==0, float("-inf"))
        weights = F.softmax(scores , dim=-1)
        weights = self.dropout(weights)
        output = torch.einsum("bhst,bhtd->bhsd",weights,v)

        output = output.transpose(1,2).contiguous()
        output = output.view(B,S,self.d_model)

        output = self.w_o(output)
        return output , weights

if __name__ == "__main__":

    B, S, D = 2, 16, 512
    H = 8

    x = torch.randn(B, S, D)

    mha = CombinedQKV_Einsum(d_model=D, n_heads=H)

    out, attn = mha(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
    print("Attn shape  :", attn.shape)
