import math


class RotaryEmbedding:
    """Rotary position embedding for list tensors shaped [batch][seq][heads][dim]."""

    def __init__(self, head_dim, max_seq_len=2048, theta=10000.0):
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.cos, self.sin = self.precompute_freqs(head_dim, max_seq_len, theta)

    def precompute_freqs(self, head_dim, max_seq_len, theta):
        inv_freqs = [
            1.0 / (theta ** (i / head_dim))
            for i in range(0, head_dim, 2)
        ]
        cos = []
        sin = []
        for position in range(max_seq_len):
            half_angles = [position * freq for freq in inv_freqs]
            angles = half_angles + half_angles
            cos.append([math.cos(angle) for angle in angles])
            sin.append([math.sin(angle) for angle in angles])
        return cos, sin

    def rotate_half(self, x):
        half = len(x) // 2
        return [-value for value in x[half:]] + x[:half]

    def _rotate_vector(self, x, position):
        rotated = self.rotate_half(x)
        return [
            x[i] * self.cos[position][i] + rotated[i] * self.sin[position][i]
            for i in range(self.head_dim)
        ]

    def _apply(self, x):
        return [
            [
                [self._rotate_vector(x[b][pos][h], pos) for h in range(len(x[b][pos]))]
                for pos in range(len(x[b]))
            ]
            for b in range(len(x))
        ]

    def forward(self, xq, xk):
        return self._apply(xq), self._apply(xk)

    __call__ = forward
