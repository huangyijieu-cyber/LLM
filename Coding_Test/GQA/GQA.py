import torch.nn as nn


class GroupQueryAttention(nn.Module):
    def __init__(self, model_dim, num_heads, num_kv_heads, dropout_p=0.0):
        super().__init__()
        assert model_dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        self.num_rep = num_heads // num_kv_heads
        self.w_q = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout_p)

    def repeat_kv(self, x, n_rep):
        """将 KV 头复制为与 Q 头数一致。"""
        raise NotImplementedError("请实现 GroupQueryAttention.repeat_kv")

    def forward(self, x, mask=None):
        """实现与 Examples/attention/GroupQueryAttention.py 一致的前向计算。"""
        raise NotImplementedError("请实现 GroupQueryAttention.forward")
