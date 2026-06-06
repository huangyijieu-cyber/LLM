import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = self.precompute_freqs(head_dim, max_seq_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def precompute_freqs(self, head_dim, max_seq_len, theta):
        """预计算形状为 [max_seq_len, head_dim] 的 cos 和 sin。"""
        raise NotImplementedError("请实现 RotaryEmbedding.precompute_freqs")

    def forward(self, xq, xk):
        """对 xq 和 xk 应用旋转位置编码，并返回二元组。"""
        raise NotImplementedError("请实现 RotaryEmbedding.forward")
