import math

from ...common import (
    combine_heads,
    concat_last,
    heads_seq_to_seq_heads,
    linear,
    scaled_dot_product_attention,
    seq_heads_to_heads_seq,
    split_last,
    unflatten_last,
)
from ..position.RotaryEmbedding import RotaryEmbedding


class MultiLatentAttention:
    """DeepSeek-style latent attention using explicit low-rank projections."""

    def __init__(self, model_dim, num_heads, head_dim, latent_dim, rope_dim, dropout_p=0.0,
                 kv_down_proj=None, kv_up_proj=None, q_down_proj=None, q_up_proj=None, o_proj=None):
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.dropout_p = dropout_p
        kv_width = num_heads * (head_dim + rope_dim + head_dim)
        q_width = num_heads * (head_dim + rope_dim)
        self.kv_down_proj = kv_down_proj if kv_down_proj is not None else [
            [0.0] * model_dim for _ in range(latent_dim)
        ]
        self.kv_up_proj = kv_up_proj if kv_up_proj is not None else [
            [0.0] * latent_dim for _ in range(kv_width)
        ]
        self.q_down_proj = q_down_proj if q_down_proj is not None else [
            [0.0] * model_dim for _ in range(latent_dim)
        ]
        self.q_up_proj = q_up_proj if q_up_proj is not None else [
            [0.0] * latent_dim for _ in range(q_width)
        ]
        self.o_proj = o_proj if o_proj is not None else [
            [0.0] * (num_heads * head_dim) for _ in range(model_dim)
        ]
        self.rope = RotaryEmbedding(head_dim=rope_dim)

    def forward(self, x, mask=None):
        kv_latent = linear(x, self.kv_down_proj)
        kv_full = unflatten_last(
            linear(kv_latent, self.kv_up_proj),
            self.num_heads,
            self.head_dim + self.rope_dim + self.head_dim,
        )
        k_content, k_rope, v_content = split_last(
            kv_full,
            [self.head_dim, self.rope_dim, self.head_dim],
        )

        q_latent = linear(x, self.q_down_proj)
        q_full = unflatten_last(
            linear(q_latent, self.q_up_proj),
            self.num_heads,
            self.head_dim + self.rope_dim,
        )
        q_content, q_rope = split_last(q_full, [self.head_dim, self.rope_dim])

        q_rope, k_rope = self.rope(q_rope, k_rope)
        q = seq_heads_to_heads_seq(concat_last(q_content, q_rope))
        k = seq_heads_to_heads_seq(concat_last(k_content, k_rope))
        v = seq_heads_to_heads_seq(v_content)

        attended, _ = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            scale=math.sqrt(self.head_dim + self.rope_dim),
        )
        seq_heads = heads_seq_to_seq_heads(attended)
        merged = combine_heads(seq_heads_to_heads_seq(seq_heads))
        return linear(merged, self.o_proj)

    __call__ = forward
