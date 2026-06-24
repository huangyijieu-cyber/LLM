import torch
from torch import nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, model_dim, num_experts, top_k):
        super().__init__()
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(model_dim, num_experts, bias = False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, 4 * model_dim),
                nn.ReLU(),
                nn.Linear(4 * model_dim, model_dim)
            )
            for _ in range(num_experts)
        ])
    def forward(self, x):
        batch_size, seq_len, model_dim = x.shape
        x_flat = x.view(batch_size * seq_len, model_dim)
        logit_gate = self.router(x_flat)
        weight, indices = torch.topk(logit_gate, self.top_k, dim = -1)
        weight = F.softmax(weight, dim = -1)
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (i == indices)
            token_indices, topk_pos = torch.where(mask)
            expert_input = x_flat[token_indices]
            expert_output = expert(expert_input)
            expert_weight = weight[token_indices, topk_pos]
            weighted_expert_output = expert_weight.unsqueeze(-1) * expert_output
            output.index_add_(0, token_indices, weighted_expert_output)
        return output.view(batch_size, seq_len, model_dim)