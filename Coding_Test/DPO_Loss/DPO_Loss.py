def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
    label_smoothing=0.0,
):
    """实现与 Examples/loss/DPOLoss.py 一致的 DPO 损失。"""
    raise NotImplementedError("请实现 dpo_loss")
