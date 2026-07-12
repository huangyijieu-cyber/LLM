from ...common import cross_entropy_from_logits


class PretrainLoss:
    """Next-token prediction loss with an optional ignore index."""

    def __init__(self, ignore_index=-100):
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        flat_logits = []
        flat_labels = []
        for batch_logits, batch_labels in zip(logits, labels):
            flat_logits.extend(batch_logits[:-1])
            flat_labels.extend(batch_labels[1:])
        return cross_entropy_from_logits(flat_logits, flat_labels, ignore_index=self.ignore_index)

    __call__ = forward
