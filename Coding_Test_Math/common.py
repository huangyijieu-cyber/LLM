import math
from copy import deepcopy


def clone(x):
    return deepcopy(x)


def is_list(x):
    return isinstance(x, list)


def is_vector(x):
    return isinstance(x, list) and (not x or not isinstance(x[0], list))


def shape(x):
    dims = []
    value = x
    while isinstance(value, list):
        dims.append(len(value))
        value = value[0] if value else None
    return tuple(dims)


def zeros(dims):
    if not dims:
        return 0.0
    return [zeros(dims[1:]) for _ in range(dims[0])]


def zeros_like(x):
    if isinstance(x, list):
        return [zeros_like(item) for item in x]
    return 0.0


def flatten(x):
    if isinstance(x, list):
        out = []
        for item in x:
            out.extend(flatten(item))
        return out
    return [x]


def map_nested(fn, x):
    if isinstance(x, list):
        return [map_nested(fn, item) for item in x]
    return fn(x)


def zip_map(fn, a, b):
    if isinstance(a, list):
        return [zip_map(fn, x, y) for x, y in zip(a, b)]
    return fn(a, b)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def mean_all(x):
    values = flatten(x)
    return mean(values)


def variance(values, unbiased=False):
    values = list(values)
    if not values:
        return float("nan")
    if unbiased and len(values) <= 1:
        return float("nan")
    mu = mean(values)
    denom = len(values) - 1 if unbiased else len(values)
    return sum((v - mu) ** 2 for v in values) / denom


def std(values, unbiased=False):
    return math.sqrt(variance(values, unbiased=unbiased))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def add_vectors(a, b):
    return [x + y for x, y in zip(a, b)]


def add_scaled(a, b, scale):
    return [x + scale * y for x, y in zip(a, b)]


def scale_vector(x, scale):
    return [scale * v for v in x]


def matvec(weight, x, bias=None):
    if bias is None:
        bias = [0.0] * len(weight)
    return [dot(row, x) + bias[i] for i, row in enumerate(weight)]


def linear(x, weight, bias=None):
    if is_vector(x):
        return matvec(weight, x, bias)
    return [linear(item, weight, bias) for item in x]


def relu_scalar(x):
    return x if x > 0.0 else 0.0


def relu(x):
    return map_nested(relu_scalar, x)


def sigmoid_scalar(x):
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def silu_scalar(x):
    return x * sigmoid_scalar(x)


def silu(x):
    return map_nested(silu_scalar, x)


def clamp_scalar(x, lo, hi):
    return min(max(x, lo), hi)


def softmax_1d(values):
    values = list(values)
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    denom = sum(exps)
    return [v / denom for v in exps]


def softmax_last(x):
    if is_vector(x):
        return softmax_1d(x)
    return [softmax_last(item) for item in x]


def logsumexp_1d(values):
    max_value = max(values)
    return max_value + math.log(sum(math.exp(v - max_value) for v in values))


def log_softmax_1d(values):
    lse = logsumexp_1d(values)
    return [v - lse for v in values]


def log_softmax(logits):
    if is_vector(logits):
        return log_softmax_1d(logits)
    return [log_softmax(item) for item in logits]


def log_sigmoid(x):
    if x >= 0.0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def cross_entropy_from_logits(flat_logits, flat_labels, ignore_index=-100):
    losses = []
    for row, label in zip(flat_logits, flat_labels):
        if label == ignore_index:
            continue
        losses.append(-log_softmax_1d(row)[label])
    return mean(losses) if losses else 0.0


def split_heads(x, num_heads):
    batch_size = len(x)
    seq_len = len(x[0]) if batch_size else 0
    model_dim = len(x[0][0]) if seq_len else 0
    head_dim = model_dim // num_heads
    out = []
    for b in range(batch_size):
        heads = []
        for h in range(num_heads):
            start = h * head_dim
            end = start + head_dim
            heads.append([x[b][s][start:end] for s in range(seq_len)])
        out.append(heads)
    return out


def combine_heads(x):
    batch_size = len(x)
    num_heads = len(x[0]) if batch_size else 0
    seq_len = len(x[0][0]) if num_heads else 0
    out = []
    for b in range(batch_size):
        rows = []
        for s in range(seq_len):
            row = []
            for h in range(num_heads):
                row.extend(x[b][h][s])
            rows.append(row)
        out.append(rows)
    return out


def unflatten_last(x, outer, inner):
    if is_vector(x):
        return [x[i * inner:(i + 1) * inner] for i in range(outer)]
    return [unflatten_last(item, outer, inner) for item in x]


def split_last(x, sizes):
    if is_vector(x):
        out = []
        start = 0
        for size in sizes:
            out.append(x[start:start + size])
            start += size
        return out
    parts = [split_last(item, sizes) for item in x]
    return [[part[i] for part in parts] for i in range(len(sizes))]


def seq_heads_to_heads_seq(x):
    batch_size = len(x)
    seq_len = len(x[0]) if batch_size else 0
    num_heads = len(x[0][0]) if seq_len else 0
    return [
        [[x[b][s][h] for s in range(seq_len)] for h in range(num_heads)]
        for b in range(batch_size)
    ]


def heads_seq_to_seq_heads(x):
    batch_size = len(x)
    num_heads = len(x[0]) if batch_size else 0
    seq_len = len(x[0][0]) if num_heads else 0
    return [
        [[x[b][h][s] for h in range(num_heads)] for s in range(seq_len)]
        for b in range(batch_size)
    ]


def concat_last(a, b):
    if is_vector(a):
        return a + b
    return [concat_last(x, y) for x, y in zip(a, b)]


def mask_at(mask, b, h, i, j):
    if mask is None:
        return 1
    dims = shape(mask)
    bi = 0 if dims[0] == 1 else b
    hi = 0 if dims[1] == 1 else h
    ii = 0 if dims[2] == 1 else i
    jj = 0 if dims[3] == 1 else j
    return mask[bi][hi][ii][jj]


def scaled_dot_product_attention(q, k, v, mask=None, scale=None):
    batch_size = len(q)
    num_heads = len(q[0]) if batch_size else 0
    q_len = len(q[0][0]) if num_heads else 0
    k_len = len(k[0][0]) if num_heads else 0
    head_dim = len(q[0][0][0]) if q_len else 0
    value_dim = len(v[0][0][0]) if k_len else 0
    divisor = scale if scale is not None else math.sqrt(head_dim)

    outputs = []
    all_weights = []
    for b in range(batch_size):
        batch_outputs = []
        batch_weights = []
        for h in range(num_heads):
            head_outputs = []
            head_weights = []
            for i in range(q_len):
                scores = []
                for j in range(k_len):
                    score = dot(q[b][h][i], k[b][h][j]) / divisor
                    if mask_at(mask, b, h, i, j) == 0:
                        score = -1e9
                    scores.append(score)
                weights = softmax_1d(scores)
                out_vec = []
                for d in range(value_dim):
                    out_vec.append(sum(weights[j] * v[b][h][j][d] for j in range(k_len)))
                head_outputs.append(out_vec)
                head_weights.append(weights)
            batch_outputs.append(head_outputs)
            batch_weights.append(head_weights)
        outputs.append(batch_outputs)
        all_weights.append(batch_weights)
    return outputs, all_weights
