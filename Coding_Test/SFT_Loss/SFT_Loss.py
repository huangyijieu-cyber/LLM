import torch.nn as nn


class SFTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, prompt_lengths):
        """屏蔽 prompt，执行 next-token shift，并计算交叉熵。"""
        raise NotImplementedError("请实现 SFTLoss.forward")
