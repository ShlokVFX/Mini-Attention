import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class sdpa(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q, k, v, mask = None):
        d_k = q.size(-1) #seq len
        scores = q @ k.transpose(-2,-1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0 , float("-inf"))
        weights = F.softmax(scores , dim=-1)
        weights = self.dropout(weights)
        out = torch.bmm(weights , v)

        return out , weights
    
if __name__ == "__main__":
    B,S,D = 2,4,8

    q = torch.randn(B,S,D)
    k = torch.randn(B,S,D)
    v = torch.randn(B,S,D)

    attn = sdpa()
    out , w = attn(q,k,v)

    print("output shape" , out.shape)
    print("weight shape" , w.shape)