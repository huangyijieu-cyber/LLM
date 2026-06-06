import torch.nn as nn
import torch
from torch.nn import functional as F
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q, k, v, mask=None):
        """
        q/k/v: [batch_size, num_heads, seq_len, head_dim]
        返回 (output, attn_weights)。
        """
        d_model = q.shape[-1]
        score = q @ k.transpose(-1, -2) / (d_model ** 0.5)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        atten = F.softmax(score, dim=-1)
        atten = self.dropout(atten)
        return atten @ v, atten
        raise NotImplementedError("请实现 ScaledDotProductAttention.forward")
