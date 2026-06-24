import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=1.0, dropout=0.0):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.weight.requires_grad = False
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.alpha = alpha
        self.rank = rank
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """按照标准答案初始化 A，并将 B 初始化为零。"""
        nn.init.kaiming_uniform_(self.lora_a.weight, a = math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        #raise NotImplementedError("请实现 LoRALinear.reset_parameters")

    def forward(self, x):
        """实现原始线性输出与 LoRA 增量输出之和。"""
        with torch.no_grad():
            original_output = self.weight(x)
        lora_output = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return original_output + lora_output
        raise NotImplementedError("请实现 LoRALinear.forward")
