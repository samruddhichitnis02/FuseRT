# FusedResidualLayerNorm: plugin interface (v1)

Computes `out = LayerNorm(hidden_state + residual) * gamma + beta` in one pass,
without writing `hidden_state + residual` to global memory.

## Identity
- Plugin name: `FusedResidualLayerNorm`
- Version: `1`
- Namespace: `fusert`
- TensorRT interface: `IPluginV3` with `IPluginCreatorV3One`

## Tensors
| Role | Name | Shape | Type |
|---|---|---|---|
| input 0 | hidden_state | [batch, seq, H] | FP32 |
| input 1 | residual | [batch, seq, H], identical to input 0 (no broadcast) | FP32 |
| input 2 | gamma | [H] | FP32 |
| input 3 | beta | [H] | FP32 |
| output 0 | normalized | [batch, seq, H], same as input 0 | FP32 |

- batch and seq may be dynamic. H must be a fixed number.
- All tensors use the plain row-major (linear) layout.
- No workspace memory is needed.

## Attributes (set at build time)
| Name | Type | Meaning |
|---|---|---|
| epsilon | float32 | LayerNorm epsilon (BERT-base: 1e-12) |
| hidden_size | int32 | H. Must equal the last dimension of input 0 and the length of gamma and beta. |

## Data types
- v1: FP32 only. All four inputs and the output must have the same type.
- Step 8 adds FP16 for hidden_state, residual and output, with FP32 as the reference.
- Any other combination is reported as unsupported to TensorRT.

## Checks at build time
The plugin rejects the node, so the fallback model is used, when:
- hidden_size differs from the last dimension or from the length of gamma or beta
- hidden_state and residual shapes differ
- H is outside the range the kernel supports (stated after Step 4)

## Serialization
Saved inside the engine file: `epsilon` (float32), `hidden_size` (int32).
Nothing else is stored.

## How it appears in the ONNX graph
- op type `FusedResidualLayerNorm`, with attributes `epsilon` and `hidden_size`
- inputs in the order: hidden_state, residual, gamma, beta
- To be confirmed in Step 5: the extra attributes the TensorRT ONNX parser needs
  to find the plugin (namespace and version).

## Fallback
FuseRT keeps two outputs:
- `bert_opt.onnx`: standard operations only (current output, always valid)
- `bert_plugin.onnx`: same graph with the plugin node
The build script loads the plugin library first. If it is missing or the plugin is
rejected, it builds from `bert_opt.onnx`.

## Reference implementation
`fusert/reference_residual_ln.py` (NumPy, float64 accumulation) is the trusted
reference for every correctness check of the kernel.