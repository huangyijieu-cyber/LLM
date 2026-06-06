import torch.nn as nn


class MoE(nn.Module):
    def __init__(self, model_dim, num_experts, top_k):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(model_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim * 4),
                nn.ReLU(),
                nn.Linear(model_dim * 4, model_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """实现 Top-k 路由、专家计算和加权融合。"""
        raise NotImplementedError("请实现 MoE.forward")
