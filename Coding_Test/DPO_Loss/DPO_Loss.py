import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
    label_smoothing=0.0,
):
    """实现与 Examples/loss/DPOLoss.py 一致的 DPO 损失。"""
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps
    logits = chosen_ratio - rejected_ratio
    dpo_loss = -F.logsigmoid(beta * logits).mean()
    if label_smoothing > 0:
        inverse_loss = -F.logsigmoid(-beta * logits).mean()
        dpo_loss = (1 - label_smoothing) * dpo_loss + label_smoothing * inverse_loss
    return dpo_loss
    raise NotImplementedError("请实现 dpo_loss")
