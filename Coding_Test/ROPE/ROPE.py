import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len = 2048, theta = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = self.precompute(max_seq_len, theta)
        self.register_buffer("cos", cos, persistent = False)
        self.register_buffer("sin", sin, persistent = False)
    def precompute(self, max_seq_len, theta):
        inv_freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
        t = torch.arange(max_seq_len, device = inv_freqs.device, dtype = torch.float32)
        angles = torch.outer(t, inv_freqs)
        angles = torch.cat((angles, angles), dim = -1)
        return angles.cos(), angles.sin()
    def forward(self, xq, xk):
        seq_len = xq.shape[1]
        cos = self.cos[:seq_len].view(1, seq_len, 1, self.head_dim)
        sin = self.sin[:seq_len].view(1, seq_len, 1, self.head_dim)
        def rotate_half(x):
            x1, x2 = torch.chunk(x, 2, dim = -1)
            return torch.cat((-x2, x1), dim = -1)
        rotated_xq = xq * cos + rotate_half(xq) * sin
        rotated_xk = xk * cos + rotate_half(xk) * sin
        return rotated_xq, rotated_xk