from ...common import cross_entropy_from_logits, log_softmax, mean, softmax_last


def softmax(logits):
    """Numerically stable softmax over the last dimension."""
    return softmax_last(logits)


def cross_entropy_loss(logits, targets):
    """Mean cross entropy for 2D logits and 1D integer targets."""
    return cross_entropy_from_logits(logits, targets)


def KL_divergence(p_logits, q_logits):
    """Mean KL(P || Q) where P and Q are induced by logits."""
    p_probs = softmax(p_logits)
    p_logs = log_softmax(p_logits)
    q_logs = log_softmax(q_logits)
    per_row = []
    for probs, p_log_row, q_log_row in zip(p_probs, p_logs, q_logs):
        per_row.append(sum(p * (lp - lq) for p, lp, lq in zip(probs, p_log_row, q_log_row)))
    return mean(per_row)
