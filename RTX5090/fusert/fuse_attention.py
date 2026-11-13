import sys
import json
import onnx
import onnx_graphsurgeon as gs


def load_with_shapes(path):
    model = onnx.load(path)

    try:
        model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception:
        model = onnx.shape_inference.infer_shapes(model)

    return gs.import_onnx(model)


def find_upstream_matmul(tensor, max_depth=12):
    """Walk backward from a tensor until one MatMul is found."""
    visited = set()

    def walk(current, depth):
        if depth > max_depth or id(current) in visited:
            return None

        visited.add(id(current))

        if len(current.inputs) != 1:
            return None

        node = current.inputs[0]

        if node.op == "MatMul":
            return node

        found = []

        for input_tensor in node.inputs:
            if isinstance(input_tensor, gs.Constant):
                continue

            result = walk(input_tensor, depth + 1)

            if result is not None:
                found.append(result)

        unique = {id(node): node for node in found}

        if len(unique) == 1:
            return next(iter(unique.values()))

        return None

    return walk(tensor, 0)


def find_downstream_matmul(tensor, max_depth=12):
    """Walk forward from a tensor until one MatMul is found."""
    visited = set()

    def walk(current, depth):
        if depth > max_depth or id(current) in visited:
            return None

        visited.add(id(current))

        found = []

        for node in current.outputs:
            if node.op == "MatMul":
                found.append(node)
                continue

            for output_tensor in node.outputs:
                result = walk(output_tensor, depth + 1)

                if result is not None:
                    found.append(result)

        unique = {id(node): node for node in found}

        if len(unique) == 1:
            return next(iter(unique.values()))

        return None

    return walk(tensor, 0)


def find_matches(graph):
    matches = []
    skipped = []

    for node in graph.nodes:
        if node.op != "Softmax":
            continue

        qk_matmul = find_upstream_matmul(node.inputs[0])

        value_matmul = None
        for output_tensor in node.outputs:
            value_matmul = find_downstream_matmul(output_tensor)

            if value_matmul is not None:
                break

        if qk_matmul is None:
            skipped.append("Softmax has no unique upstream Q x K^T MatMul")
            continue

        if value_matmul is None:
            skipped.append("Softmax has no unique downstream scores x V MatMul")
            continue

        matches.append(
            {
                "q_times_k_node": qk_matmul.name,
                "softmax_node": node.name,
                "scores_times_v_node": value_matmul.name,
            }
        )

    return matches, skipped


if __name__ == "__main__":
    onnx_path = sys.argv[1]
    out_json = sys.argv[2]

    graph = load_with_shapes(onnx_path)
    matches, skipped = find_matches(graph)

    print(f"ONNX model: {onnx_path}")
    print(f"attention matches: {len(matches)}")
    print(f"skipped softmax nodes: {len(skipped)}")

    for index, match in enumerate(matches):
        print()
        print(f"attention block {index}")
        print(f"  Q x K^T:     {match['q_times_k_node']}")
        print(f"  Softmax:     {match['softmax_node']}")
        print(f"  scores x V:  {match['scores_times_v_node']}")

    with open(out_json, "w") as f:
        json.dump(
            {
                "onnx_model": onnx_path,
                "matches": matches,
                "skipped": skipped,
            },
            f,
            indent=2,
        )

    print()
    print(f"saved JSON results to {out_json}")