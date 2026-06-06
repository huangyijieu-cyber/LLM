def compute_grpo_advantages(rewards):
    """使用组内均值和样本标准差计算优势。"""
    raise NotImplementedError("请实现 compute_grpo_advantages")


def grpo_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2, beta=0.01, ref_kl=None):
    """实现带可选 KL 惩罚的 GRPO Loss。"""
    raise NotImplementedError("请实现 grpo_loss")


def compute_kl_penalty(log_probs, ref_log_probs):
    """实现 Schulman KL 估计器。"""
    raise NotImplementedError("请实现 compute_kl_penalty")
