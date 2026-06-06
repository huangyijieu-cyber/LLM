import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, model_dim, intermediate_dim):
        super().__init__()
        self.model_dim = model_dim
        self.intermediate_dim = intermediate_dim
        self.w_up = nn.Linear(model_dim, intermediate_dim)
        self.w_down = nn.Linear(intermediate_dim, model_dim)

    def forward(self, x):
        """实现 Linear -> ReLU -> Linear。"""
        raise NotImplementedError("请实现 FFN.forward")
