import torch

def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_epsilon = 0.2):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    loss = -torch.mean(torch.min(surrogate1, surrogate2))
    return loss