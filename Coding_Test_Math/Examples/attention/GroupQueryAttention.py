from ...common import combine_heads, linear, scaled_dot_product_attention, split_heads


class GroupQueryAttention:
    """Grouped-query attention where several query heads share one KV head."""

    def __init__(self, model_dim, num_heads, num_kv_heads, dropout_p=0.0,
                 w_q=None, w_k=None, w_v=None, w_o=None, b_o=None):
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        self.num_rep = num_heads // num_kv_heads
        self.dropout_p = dropout_p
        self.w_q = w_q if w_q is not None else [[0.0] * model_dim for _ in range(num_heads * self.head_dim)]
        self.w_k = w_k if w_k is not None else [[0.0] * model_dim for _ in range(num_kv_heads * self.head_dim)]
        self.w_v = w_v if w_v is not None else [[0.0] * model_dim for _ in range(num_kv_heads * self.head_dim)]
        self.w_o = w_o if w_o is not None else [[0.0] * model_dim for _ in range(model_dim)]
        self.b_o = b_o if b_o is not None else [0.0] * model_dim

    def repeat_kv(self, x, n_rep):
        if n_rep == 1:
            return x
        out = []
        for batch in x:
            repeated_heads = []
            for head in batch:
                for _ in range(n_rep):
                    repeated_heads.append(head)
            out.append(repeated_heads)
        return out

    def forward(self, x, mask=None):
        q = split_heads(linear(x, self.w_q), self.num_heads)
        k = split_heads(linear(x, self.w_k), self.num_kv_heads)
        v = split_heads(linear(x, self.w_v), self.num_kv_heads)
        k = self.repeat_kv(k, self.num_rep)
        v = self.repeat_kv(v, self.num_rep)
        attended, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        merged = combine_heads(attended)
        return linear(merged, self.w_o, self.b_o)

    __call__ = forward
