import torch.nn as nn
import torch
import torch.nn.functional as F

class SFTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, prompt_lengths):
        """
        前向传播

        Args:
            logits: 模型输出的未归一化对数概率 [batch_size, seq_len, vocab_size]
            labels: 真实词元索引 [batch_size, seq_len]
            prompt_lengths: 每个样本的 prompt 长度 [batch_size]

        Returns:
            loss: 标量损失值
        """
        batch_size, seq_len, vocab_size = logits.shape
        masked_labels = labels.clone()
        for batch_idx, prompt_length in enumerate(prompt_lengths):
            masked_labels[batch_idx, :prompt_length] = -100
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = masked_labels[:, 1:].contiguous()
        flat_logits = shifted_logits.view(-1, vocab_size)
        flat_labels = shifted_labels.view(-1)
        loss = F.cross_entropy(flat_logits, flat_labels, ignore_index = -100)
        return loss

        raise NotImplementedError("请实现 SFTLoss.forward")
