from ...common import scaled_dot_product_attention


class ScaledDotProductAttention:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V."""

    def __init__(self, dropout_p=0.0):
        self.dropout_p = dropout_p

    def forward(self, q, k, v, mask=None):
        return scaled_dot_product_attention(q, k, v, mask=mask)

    __call__ = forward
