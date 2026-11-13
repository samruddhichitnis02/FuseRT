import numpy as np


def residual_layernorm(hidden_state, residual, gamma, beta, epsilon):
    """Reference for FusedResidualLayerNorm. Math is done in float64."""
    x = hidden_state.astype(np.float64) + residual.astype(np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + epsilon) * gamma.astype(np.float64) + beta.astype(np.float64)
    return y.astype(hidden_state.dtype)