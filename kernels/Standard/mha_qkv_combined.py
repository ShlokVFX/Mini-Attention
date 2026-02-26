import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CombinedQKV(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(d_model, 3*d_model)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):

        B,S, _ = x.shape

        #linear proj
        qkv = self.qkv(x)

        #head split
        qkv = qkv.view(B, S, 3, self.n_heads, self.d_h).permute(2,0,3,1,4) 
        #(3,B,H,S,Dh)

        #qkv unbind for sdpa
        q,k,v=qkv.unbind(0)

        #sdpa

        scores = q @ k.transpose(-2,-1)
        scores /= math.sqrt(self.d_h)

        if mask is not None:
            mask = mask.unsqueeze(1) #(B,1,S,S)
            scores = scores.masked_fill(mask==0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v #(B,H,S,Dh)

        #heads concat
        output = output.transpose(1,2).contiguous()
        output = output.view(B,S,self.d_model)

        #output proj
        output = self.o(output)

        return output , weights

if __name__ == "__main__":
    B, S, D = 2, 16, 512
    H = 8

    x = torch.randn(B, S, D)

    mha = CombinedQKV(d_model=D, n_heads=H)

    out, attn = mha(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
    print("Attn shape  :", attn.shape)