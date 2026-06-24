import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(model_dim))

    def _norm(self, x):
        """实现 RMS 归一化核心计算。"""
        var = torch.mean(x ** 2, dim = -1, keepdims = True)
        norm = x / (var + self.eps) ** 0.5
        return norm

        raise NotImplementedError("请实现 RMSNorm._norm")

    def forward(self, x):
        """实现与 Examples/normalization/RMSNorm.py 一致的前向计算。"""
        return self._norm(x) * self.gamma
        raise NotImplementedError("请实现 RMSNorm.forward")
