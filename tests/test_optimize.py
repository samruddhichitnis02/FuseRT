import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import helper, TensorProto, numpy_helper
from fusert import optimize as opt


def _build(nodes, inputs, outputs, inits=(), opset=17):
    g = helper.make_graph(nodes, "t", inputs, outputs, list(inits))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    return gs.import_onnx(m)


def _assert_valid(graph):
    onnx.checker.check_model(gs.export_onnx(graph))


def test_inline_constants():
    value = helper.make_tensor("v", TensorProto.FLOAT, [3], [2.0, -1.0, 0.5])
    graph = _build(
        [
            helper.make_node("Constant", [], ["c"], value=value),
            helper.make_node("Add", ["x", "c"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
    )

    opt.inline_constants(graph)

    assert [n.op for n in graph.nodes].count("Constant") == 0
    assert len(graph.nodes) == 1

    (add,) = graph.nodes
    assert isinstance(add.inputs[1], gs.Constant)
    assert np.array_equal(add.inputs[1].values, np.array([2.0, -1.0, 0.5], np.float32))
    _assert_valid(graph)

def test_inline_constants_value_float():
    graph = _build(
        [
            helper.make_node("Constant", [], ["c"], value_float=3.0),
            helper.make_node("Mul", ["x", "c"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )

    opt.inline_constants(graph)

    assert [n.op for n in graph.nodes].count("Constant") == 0

    (mul,) = graph.nodes
    assert isinstance(mul.inputs[1], gs.Constant)
    assert float(np.asarray(mul.inputs[1].values).reshape(-1)[0]) == 3.0
    _assert_valid(graph)

def test_inline_constants_keep_graph_output():
    """Deleting this node would leave the graph output unproduced."""
    graph = _build(
        [
            helper.make_node(
                "Constant",
                [],
                ["k"],
                value=helper.make_tensor("v", TensorProto.FLOAT, [2], [1.0, 2.0]),
            )
        ],
        [],
        [helper.make_tensor_value_info("k", TensorProto.FLOAT, [2])],
    )

    opt.inline_constants(graph)

    assert [n.op for n in graph.nodes].count("Constant") == 1
    _assert_valid(graph)


def test_inline_constants_keep_sparse_constant():
    """graphsurgeon cannot read sparse_value, so the node must be left alone."""
    indices = numpy_helper.from_array(np.array([[0], [2]], np.int64))  # (nnz, rank)
    values = numpy_helper.from_array(np.array([1.0, 3.0], np.float32))
    sparse = helper.make_sparse_tensor(values, indices, [4])
    graph = _build(
        [helper.make_node("Constant", [], ["c"], sparse_value=sparse)],
        [],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, [4])],
    )

    opt.inline_constants(graph)

    assert [n.op for n in graph.nodes].count("Constant") == 1
    _assert_valid(graph)


def test_inline_constants_rewires_every_consumer():
    """One constant read by two nodes must be rewired for both of them."""
    graph = _build(
        [
            helper.make_node(
                "Constant",
                [],
                ["c"],
                value=helper.make_tensor("v", TensorProto.FLOAT, [1], [2.0]),
            ),
            helper.make_node("Add", ["x", "c"], ["a"]),
            helper.make_node("Mul", ["x", "c"], ["b"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [2]),
        ],
    )

    opt.inline_constants(graph)

    assert [n.op for n in graph.nodes].count("Constant") == 0

    add = next(n for n in graph.nodes if n.op == "Add")
    mul = next(n for n in graph.nodes if n.op == "Mul")
    assert isinstance(add.inputs[1], gs.Constant)
    assert isinstance(mul.inputs[1], gs.Constant)
    assert add.inputs[1] is mul.inputs[1]
    _assert_valid(graph)