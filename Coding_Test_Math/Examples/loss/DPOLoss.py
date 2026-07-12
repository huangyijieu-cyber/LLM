from ...common import flatten, log_sigmoid, mean


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1, label_smoothing=0.0):
    """Direct Preference Optimization loss."""
    chosen = flatten(policy_chosen_logps)
    rejected = flatten(policy_rejected_logps)
    ref_chosen = flatten(ref_chosen_logps)
    ref_rejected = flatten(ref_rejected_logps)
    losses = []
    inverse_losses = []
    for pc, pr, rc, rr in zip(chosen, rejected, ref_chosen, ref_rejected):
        logits = (pc - rc) - (pr - rr)
        losses.append(-log_sigmoid(beta * logits))
        inverse_losses.append(-log_sigmoid(-beta * logits))
    base = mean(losses)
    if label_smoothing > 0.0:
        return (1.0 - label_smoothing) * base + label_smoothing * mean(inverse_losses)
    return base
