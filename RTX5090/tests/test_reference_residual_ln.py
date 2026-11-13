import numpy as np
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
from fusert.reference_residual_ln import residual_layernorm


def _session(h, eps, gamma, beta):
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [None, None, h])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [None, None, h])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, None, h])
    nodes = [
        helper.make_node("Add", ["a", "b"], ["s"]),
        helper.make_node("LayerNormalization", ["s", "gamma", "beta"], ["y"],
                         axis=-1, epsilon=eps),
    ]
    inits = [numpy_helper.from_array(gamma, "gamma"),
             numpy_helper.from_array(beta, "beta")]
    graph = helper.make_graph(nodes, "g", [a, b], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    return ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])


def _check(shape, eps):
    rng = np.random.default_rng(0)
    h = shape[-1]
    a = rng.normal(size=shape).astype(np.float32)
    b = rng.normal(size=shape).astype(np.float32)
    gamma = rng.normal(size=h).astype(np.float32)
    beta = rng.normal(size=h).astype(np.float32)
    got = _session(h, eps, gamma, beta).run(None, {"a": a, "b": b})[0]
    want = residual_layernorm(a, b, gamma, beta, eps)
    assert np.abs(got - want).max() < 1e-5


def test_bert_shape():
    _check((2, 5, 768), 1e-12)


def test_single_token():
    _check((1, 1, 768), 1e-12)


def test_other_hidden_size():
    _check((3, 7, 256), 1e-5)