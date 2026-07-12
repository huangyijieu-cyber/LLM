import math

from ...common import clamp_scalar, flatten, mean


def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2):
    """PPO clipped policy loss."""
    old_values = flatten(old_log_probs)
    new_values = flatten(new_log_probs)
    adv_values = flatten(advantages)
    losses = []
    for old, new, adv in zip(old_values, new_values, adv_values):
        ratio = math.exp(new - old)
        clipped_ratio = clamp_scalar(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        surrogate1 = ratio * adv
        surrogate2 = clipped_ratio * adv
        losses.append(-min(surrogate1, surrogate2))
    return mean(losses)
