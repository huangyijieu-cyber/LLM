import math

from ...common import clamp_scalar, flatten, mean, std


def compute_grpo_advantages(rewards):
    """Group-normalized advantages: (R - mean(R)) / (std(R) + eps)."""
    advantages = []
    for row in rewards:
        row_mean = mean(row)
        row_std = std(row, unbiased=True)
        advantages.append([(value - row_mean) / (row_std + 1e-8) for value in row])
    return advantages


def grpo_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2, beta=0.01, ref_kl=None):
    """GRPO clipped policy loss with an optional KL penalty."""
    old_values = flatten(old_log_probs)
    new_values = flatten(new_log_probs)
    adv_values = flatten(advantages)
    kl_values = flatten(ref_kl) if ref_kl is not None else None
    losses = []
    for idx, (old, new, adv) in enumerate(zip(old_values, new_values, adv_values)):
        ratio = math.exp(new - old)
        clipped_ratio = clamp_scalar(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate1 = ratio * adv
        surrogate2 = clipped_ratio * adv
        policy_loss = -min(surrogate1, surrogate2)
        if kl_values is not None:
            policy_loss += beta * kl_values[idx]
        losses.append(policy_loss)
    return mean(losses)


def compute_kl_penalty(log_probs, ref_log_probs):
    """Schulman-style KL estimator: exp(ref-log - log) - (ref-log - log) - 1."""
    values = []
    for log_p, ref_log_p in zip(flatten(log_probs), flatten(ref_log_probs)):
        log_ratio = ref_log_p - log_p
        values.append(math.exp(log_ratio) - log_ratio - 1.0)
    return mean(values)
