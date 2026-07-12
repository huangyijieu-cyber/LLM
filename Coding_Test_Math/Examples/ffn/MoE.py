from ...common import add_scaled, linear, relu, softmax_1d


class MoE:
    """Top-k mixture of experts with plain Python list tensors."""

    def __init__(self, model_dim, num_experts, top_k, router=None, experts=None):
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = router if router is not None else [
            [0.0] * model_dim for _ in range(num_experts)
        ]
        self.experts = experts if experts is not None else [
            {
                "w1": [[0.0] * model_dim for _ in range(4 * model_dim)],
                "b1": [0.0] * (4 * model_dim),
                "w2": [[0.0] * (4 * model_dim) for _ in range(model_dim)],
                "b2": [0.0] * model_dim,
            }
            for _ in range(num_experts)
        ]

    def _expert_forward(self, expert, token):
        hidden = relu(linear(token, expert["w1"], expert["b1"]))
        return linear(hidden, expert["w2"], expert["b2"])

    def _token_forward(self, token):
        gate_logits = linear(token, self.router)
        ranked = sorted(enumerate(gate_logits), key=lambda item: item[1], reverse=True)
        chosen = ranked[:self.top_k]
        weights = softmax_1d([score for _, score in chosen])
        output = [0.0] * self.model_dim
        for pos, (expert_idx, _) in enumerate(chosen):
            expert_output = self._expert_forward(self.experts[expert_idx], token)
            output = add_scaled(output, expert_output, weights[pos])
        return output

    def forward(self, x):
        return [[self._token_forward(token) for token in sequence] for sequence in x]

    __call__ = forward
