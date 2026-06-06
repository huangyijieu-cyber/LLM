import torch.nn as nn


class SwiGLUFFN(nn.Module):
    def __init__(self, model_dim, intermediate_dim):
        super().__init__()
        self.model_dim = model_dim
        self.intermediate_dim = intermediate_dim
        self.w_gate = nn.Linear(model_dim, intermediate_dim, bias=False)
        self.w_up = nn.Linear(model_dim, intermediate_dim, bias=False)
        self.w_down = nn.Linear(intermediate_dim, model_dim, bias=False)

    def forward(self, x):
        """实现 Down(SiLU(Gate(x)) * Up(x))。"""
        raise NotImplementedError("请实现 SwiGLUFFN.forward")
