import math
import torch


def fused_attention_reference(q, k, v, attention_mask=None):
    """
    Correct reference implementation of BERT-style scaled dot-product attention.

    q, k, v:
        [batch, heads, sequence_length, head_size]

    attention_mask:
        [batch, sequence_length]
        1 = real token
        0 = padding token
    """

    head_size = q.shape[-1]

    # Q × K^T, scaled by 1 / sqrt(head_size).
    scores = torch.matmul(q, k.transpose(-1, -2))
    scores = scores / math.sqrt(head_size)

    # Do not allow attention to padding tokens.
    if attention_mask is not None:
        invalid_positions = attention_mask[:, None, None, :] == 0
        scores = scores.masked_fill(
            invalid_positions,
            torch.finfo(scores.dtype).min,
        )

    # Convert scores into probabilities.
    probabilities = torch.softmax(scores, dim=-1)

    # Attention probabilities × V.
    output = torch.matmul(probabilities, v)

    return output