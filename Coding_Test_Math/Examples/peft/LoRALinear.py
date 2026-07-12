from ...common import linear, zip_map


class LoRALinear:
    """LoRA linear layer: y = W x + scaling * B(A(x))."""

    def __init__(self, in_features, out_features, rank=8, alpha=1.0, dropout=0.0,
                 weight=None, lora_a=None, lora_b=None):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = dropout
        self.weight = weight if weight is not None else [
            [0.0] * in_features for _ in range(out_features)
        ]
        self.lora_a = lora_a if lora_a is not None else [
            [0.0] * in_features for _ in range(rank)
        ]
        self.lora_b = lora_b if lora_b is not None else [
            [0.0] * rank for _ in range(out_features)
        ]

    def reset_parameters(self):
        self.lora_b = [[0.0] * self.rank for _ in range(self.out_features)]

    def forward(self, x):
        original_output = linear(x, self.weight)
        lora_hidden = linear(x, self.lora_a)
        lora_output = linear(lora_hidden, self.lora_b)
        scaled_lora = zip_map(lambda _, value: value * self.scaling, original_output, lora_output)
        return zip_map(lambda base, delta: base + delta, original_output, scaled_lora)

    __call__ = forward
