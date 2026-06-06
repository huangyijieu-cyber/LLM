import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(model_dim))
        self.beta = nn.Parameter(torch.zeros(model_dim))

    def forward(self, x):
        """实现与 Examples/normalization/LayerNorm.py 一致的前向计算。"""
        raise NotImplementedError("请实现 LayerNorm.forward")
