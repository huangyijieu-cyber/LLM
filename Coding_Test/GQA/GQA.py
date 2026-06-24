import torch
from torch import nn
import torch.nn.functional as F
import math
class GroupQueryAttention(nn.Module):
    def __init__(self, model_dim, num_heads, num_kv_heads, dropout_p = 0.0):
        super().__init__()
        assert model_dim % num_heads == 0, "bug"
        assert num_heads % num_kv_heads == 0, "bug"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = nn.Dropout(dropout_p)
        self.head_dim = model_dim // num_heads
        self.rep = num_heads // num_kv_heads
        self.w_q = nn.Linear(model_dim, num_heads * self.head_dim, bias = False)
        self.w_k = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias = False)
        self.w_v = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias = False)
        self.w_o = nn.Linear(model_dim, model_dim)

    def repeat(self, x):
        if self.repeat == 1:
            return x
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :]
        x = x.expand(batch_size, num_heads, self.rep, seq_len, head_dim)
        x = x.reshape(batch_size, num_heads * self.rep, seq_len, head_dim)
        return x

    def forward(self, x, mask = None):
        batch_size = x.shape[0]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = self.repeat(k)
        v = self.repeat(v)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        atten = F.softmax(scores, dim = -1)
        atten = self.dropout(atten)
        output = torch.matmul(atten, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
        return self.w_o(output)
