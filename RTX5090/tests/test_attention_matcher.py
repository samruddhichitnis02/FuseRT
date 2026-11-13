import numpy as np
import onnx_graphsurgeon as gs

from fusert.fuse_attention import find_matches


def build_attention_graph(include_value_matmul=True):
    q = gs.Variable("q", dtype=np.float32, shape=(1, 8, 64))
    k_transpose = gs.Variable(
        "k_transpose",
        dtype=np.float32,
        shape=(1, 64, 8),
    )
    v = gs.Variable("v", dtype=np.float32, shape=(1, 8, 64))

    scores = gs.Variable("scores", dtype=np.float32, shape=(1, 8, 8))
    probabilities = gs.Variable(
        "probabilities",
        dtype=np.float32,
        shape=(1, 8, 8),
    )
    output = gs.Variable("output", dtype=np.float32, shape=(1, 8, 64))

    q_times_k = gs.Node(
        op="MatMul",
        name="q_times_k",
        inputs=[q, k_transpose],
        outputs=[scores],
    )

    softmax = gs.Node(
        op="Softmax",
        name="softmax",
        inputs=[scores],
        outputs=[probabilities],
        attrs={"axis": -1},
    )

    nodes = [q_times_k, softmax]
    outputs = [probabilities]

    if include_value_matmul:
        scores_times_v = gs.Node(
            op="MatMul",
            name="scores_times_v",
            inputs=[probabilities, v],
            outputs=[output],
        )

        nodes.append(scores_times_v)
        outputs = [output]

    return gs.Graph(
        nodes=nodes,
        inputs=[q, k_transpose, v],
        outputs=outputs,
    )


def test_complete_attention_pattern_matches():
    matches, skipped = find_matches(build_attention_graph(True))

    assert len(matches) == 1
    assert len(skipped) == 0

    assert matches[0]["q_times_k_node"] == "q_times_k"
    assert matches[0]["softmax_node"] == "softmax"
    assert matches[0]["scores_times_v_node"] == "scores_times_v"


def test_incomplete_attention_pattern_is_skipped():
    matches, skipped = find_matches(build_attention_graph(False))

    assert len(matches) == 0
    assert len(skipped) == 1