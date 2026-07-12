from ...common import clone, cross_entropy_from_logits


class SFTLoss:
    """Supervised fine-tuning loss that ignores prompt tokens."""

    def __init__(self):
        pass

    def forward(self, logits, labels, prompt_lengths):
        masked_labels = clone(labels)
        for batch_idx, prompt_length in enumerate(prompt_lengths):
            for pos in range(prompt_length):
                masked_labels[batch_idx][pos] = -100

        flat_logits = []
        flat_labels = []
        for batch_logits, batch_labels in zip(logits, masked_labels):
            flat_logits.extend(batch_logits[:-1])
            flat_labels.extend(batch_labels[1:])
        return cross_entropy_from_logits(flat_logits, flat_labels, ignore_index=-100)

    __call__ = forward
