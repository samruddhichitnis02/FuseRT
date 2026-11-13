#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace fusert {

struct FusedResidualLayerNormParams {
    float epsilon;        // LayerNorm epsilon
    int32_t hidden_size;  // size of the last dimension (H)
};

enum class FusedDType : int32_t { kFloat32 = 0, kFloat16 = 1 };

// out = LayerNorm(hidden_state + residual) * gamma + beta
// hidden_state, residual, out: row-major [rows, hidden_size], rows = batch * seq
// gamma, beta: [hidden_size]
// All pointers are device pointers. Returns cudaSuccess or the launch error.
cudaError_t launch_fused_residual_layernorm(
    const void* hidden_state, const void* residual,
    const void* gamma, const void* beta, void* out,
    int64_t rows, const FusedResidualLayerNormParams& params,
    FusedDType dtype, cudaStream_t stream);

}  // namespace fusert