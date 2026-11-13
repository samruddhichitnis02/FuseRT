import numpy as np
import onnx_graphsurgeon as gs
from fusert.fuse_residual_layernorm import find_matches

SHAPE = ("batch", "seq", 8)


def build(b_kind="same", extra_consumer=False):
    a = gs.Variable("a", dtype=np.float32, shape=SHAPE)
    if b_kind == "const":
        b = gs.Constant("b", np.zeros(8, dtype=np.float32))
    elif b_kind == "broadcast":
        b = gs.Variable("b", dtype=np.float32, shape=(1, "seq", 8))
    else:
        b = gs.Variable("b", dtype=np.float32, shape=SHAPE)
    s = gs.Variable("s", dtype=np.float32, shape=SHAPE)
    y = gs.Variable("y", dtype=np.float32, shape=SHAPE)
    gamma = gs.Constant("gamma", np.ones(8, dtype=np.float32))
    beta = gs.Constant("beta", np.zeros(8, dtype=np.float32))
    nodes = [
        gs.Node("Add", inputs=[a, b], outputs=[s]),
        gs.Node("LayerNormalization", inputs=[s, gamma, beta], outputs=[y],
                attrs={"axis": -1, "epsilon": 1e-12}),
    ]
    outputs = [y]
    if extra_consumer:
        z = gs.Variable("z", dtype=np.float32, shape=SHAPE)
        nodes.append(gs.Node("Relu", inputs=[s], outputs=[z]))
        outputs.append(z)
    inputs = [a] if b_kind == "const" else [a, b]
    return gs.Graph(nodes=nodes, inputs=inputs, outputs=outputs)


def test_residual_add_matches():
    matches, skipped, total = find_matches(build("same"))
    assert total == 1 and len(matches) == 1 and not skipped


def test_bias_add_is_skipped():
    matches, skipped, _ = find_matches(build("const"))
    assert len(matches) == 0 and sum(skipped.values()) == 1


def test_broadcast_add_is_skipped():
    matches, skipped, _ = find_matches(build("broadcast"))
    assert len(matches) == 0 and sum(skipped.values()) == 1


def test_add_used_elsewhere_is_skipped():
    matches, skipped, _ = find_matches(build("same", extra_consumer=True))
    assert len(matches) == 0 and sum(skipped.values()) == 1