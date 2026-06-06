import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout_p=0.0):
        super().__init__()
        assert model_dim % num_heads == 0
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x_query, x_context, mask=None):
        """
        x_query: [batch_size, seq_len_q, model_dim]
        x_context 为 None 时执行自注意力，否则执行交叉注意力。
        """
        q = self.w_q(x_query)
        if x_context is not None:
            k = self.w_k(x_context)
            v = self.w_v(x_context)
        else:
            k = self.w_k(x_query)
            v = self.w_v(x_query)
        batch_size, seq_len, d_model = q.shape
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        atten = self.dropout(torch.softmax(scores, dim = -1))
        output = torch.matmul(atten, v).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(output)
        raise NotImplementedError("请实现 MultiHeadAttention.forward")
