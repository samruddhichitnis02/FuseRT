import sys
import collections
import numpy as np
import onnx
import onnx_graphsurgeon as gs


def load_with_shapes(path):
    """Load an ONNX file and fill in tensor shapes so we can compare them."""
    model = onnx.load(path)
    try:
        model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception:
        model = onnx.shape_inference.infer_shapes(model)
    return gs.import_onnx(model)


def _producer(t, op):
    if len(t.inputs) == 1 and t.inputs[0].op == op:
        return t.inputs[0]
    return None


def _shape_known(t):
    return t.shape is not None and all(d is not None for d in t.shape)


def _match(ln):
    """Check one LayerNormalization node. Returns (match, None) or (None, reason)."""
    x = ln.inputs[0]
    add = _producer(x, "Add")
    if add is None:
        return None, "LayerNorm input is not an Add"

    a, b = add.inputs
    if isinstance(a, gs.Constant) or isinstance(b, gs.Constant):
        return None, "Add has a constant input (bias add, not a residual)"
    if len(x.outputs) != 1:
        return None, "Add result is used by more than the LayerNorm"
    if int(ln.attrs.get("axis", -1)) != -1:
        return None, "LayerNorm axis is not the last one"

    gamma, beta = ln.inputs[1], ln.inputs[2] if len(ln.inputs) > 2 else None
    if not isinstance(gamma, gs.Constant) or not isinstance(beta, gs.Constant):
        return None, "gamma or beta is not a constant"

    if a.dtype != np.float32 or b.dtype != np.float32:
        return None, "inputs are not float32"
    if not (_shape_known(a) and _shape_known(b)):
        return None, "input shapes are not fully known"
    if list(a.shape) != list(b.shape):
        return None, f"input shapes differ (broadcast?): {list(a.shape)} vs {list(b.shape)}"

    hidden = int(gamma.values.shape[-1])
    if int(a.shape[-1]) != hidden:
        return None, "last dimension does not match gamma size"

    return dict(add=add, ln=ln, hidden_state=a, residual=b, gamma=gamma,
                beta=beta, epsilon=float(ln.attrs.get("epsilon", 1e-5)),
                hidden_size=hidden), None


def find_matches(graph):
    matches, skipped, total = [], collections.Counter(), 0
    for node in graph.nodes:
        if node.op != "LayerNormalization":
            continue
        total += 1
        m, reason = _match(node)
        if m is None:
            skipped[reason] += 1
        else:
            matches.append(m)
    return matches, skipped, total


if __name__ == "__main__":
    g = load_with_shapes(sys.argv[1])
    matches, skipped, total = find_matches(g)
    print(f"LayerNormalization nodes: {total}")
    print(f"residual + LayerNorm matches: {len(matches)}")
    for reason, n in skipped.items():
        print(f"skipped {n}: {reason}")
    for i, m in enumerate(matches[:3]):
        print(f"example {i}: shape={list(m['hidden_state'].shape)} "
              f"hidden={m['hidden_size']} eps={m['epsilon']}")