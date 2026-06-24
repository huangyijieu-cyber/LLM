import torch
from torch import nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, drop_out = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(drop_out)
    
    def forward(self, q, k, v, mask = None):
        d_model = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_model)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        atten = F.softmax(scores, dim = -1)
        atten = self.dropout(atten)
        return torch.matmul(atten, v), atten