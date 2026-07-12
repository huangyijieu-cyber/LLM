from ...common import combine_heads, linear, scaled_dot_product_attention, split_heads


class MultiHeadAttention:
    """Multi-head self/cross attention using explicit Python list operations."""

    def __init__(self, model_dim, num_heads, dropout_p=0.0,
                 w_q=None, b_q=None, w_k=None, b_k=None, w_v=None, b_v=None, w_o=None, b_o=None):
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.dropout_p = dropout_p
        self.w_q = w_q if w_q is not None else [[0.0] * model_dim for _ in range(model_dim)]
        self.b_q = b_q if b_q is not None else [0.0] * model_dim
        self.w_k = w_k if w_k is not None else [[0.0] * model_dim for _ in range(model_dim)]
        self.b_k = b_k if b_k is not None else [0.0] * model_dim
        self.w_v = w_v if w_v is not None else [[0.0] * model_dim for _ in range(model_dim)]
        self.b_v = b_v if b_v is not None else [0.0] * model_dim
        self.w_o = w_o if w_o is not None else [[0.0] * model_dim for _ in range(model_dim)]
        self.b_o = b_o if b_o is not None else [0.0] * model_dim

    def forward(self, x_query, x_context=None, mask=None):
        context = x_query if x_context is None else x_context
        q = split_heads(linear(x_query, self.w_q, self.b_q), self.num_heads)
        k = split_heads(linear(context, self.w_k, self.b_k), self.num_heads)
        v = split_heads(linear(context, self.w_v, self.b_v), self.num_heads)
        attended, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        merged = combine_heads(attended)
        return linear(merged, self.w_o, self.b_o)

    __call__ = forward
