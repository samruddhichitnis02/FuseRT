import math
import torch
import torch.nn.functional as F

from fusert.reference_fused_attention import fused_attention_reference


def test_reference_matches_pytorch_attention():
    torch.manual_seed(7)

    batch = 2
    heads = 12
    sequence_length = 8
    head_size = 64

    q = torch.randn(batch, heads, sequence_length, head_size)
    k = torch.randn(batch, heads, sequence_length, head_size)
    v = torch.randn(batch, heads, sequence_length, head_size)

    # The last two tokens of the second example are padding.
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ]
    )

    actual = fused_attention_reference(q, k, v, attention_mask)

    pytorch_mask = attention_mask[:, None, None, :].bool()

    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=pytorch_mask,
        dropout_p=0.0,
        scale=1.0 / math.sqrt(head_size),
    )

    max_abs_error = (actual - expected).abs().max().item()

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5), (
        f"maximum absolute error: {max_abs_error}"
    )