# Paper: https://arxiv.org/abs/1706.03762
# Section: Multi-Head Attention

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MHA(nn.Module):
    def __init__(self , d_model , n_heads, dropout_p=0.0):
        super().__init__()

        #model should be divisble by heads
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.dropout = nn.Dropout(dropout_p)

        #projection matrix
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self , x , mask=None):

        B , S , _ = x.shape

        #linear projection
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        #split into heads
        q = q.view(B , S , self.n_heads , self.d_h).transpose(1,2)
        k = k.view(B , S, self.n_heads,self.d_h).transpose(1,2)
        v = v.view(B, S, self.n_heads, self.d_h).transpose(1,2)

        #Scaled dot product
        scores = q @ k.transpose(-2,-1) / math.sqrt(self.d_h)
        if mask is not None:
            scores = scores.masked_fill(mask==0 , float("-inf"))
        weights = F.softmax(scores , dim=-1)
        weights = self.dropout(weights)
        output = weights @ v

        #concat heads
        output = output.transpose(1,2).contiguous()
        output = output.view(B , S , self.d_model)

        #final projection
        output = self.w_o(output)

        return output , weights
         

if __name__ == "__main__":
    B, S, D = 2, 16, 512
    H = 8

    x = torch.randn(B, S, D)

    mha = MHA(d_model=D, n_heads=H)

    out, attn = mha(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
    print("Attn shape  :", attn.shape)


