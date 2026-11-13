import sys
import json

breakdown_json = sys.argv[1]
layers_json = sys.argv[2]
out_json = sys.argv[3]

with open(breakdown_json) as f:
    breakdown = json.load(f)

with open(layers_json) as f:
    layer_file = json.load(f)

layers = {
    layer["Name"]: layer
    for layer in layer_file["Layers"]
    if isinstance(layer, dict) and "Name" in layer
}


def tensor_info(tensor):
    return {
        "name": tensor.get("Name"),
        "dimensions": tensor.get("Dimensions"),
        "datatype": tensor.get("Datatype"),
    }


selected = []

for row in breakdown["rows"]:
    category = row["category"]

    if category not in (
        "q_times_k_transpose_gemm",
        "softmax",
        "scores_times_v_gemm",
    ):
        continue

    layer = layers.get(row["name"])

    if layer is None:
        continue

    selected.append(
        {
            "category": category,
            "layer_name": row["name"],
            "time_ms": row["ms"],
            "inputs": [
                tensor_info(tensor)
                for tensor in layer.get("Inputs", [])
            ],
            "outputs": [
                tensor_info(tensor)
                for tensor in layer.get("Outputs", [])
            ],
            "metadata": layer.get("Metadata", ""),
        }
    )


print(f"selected attention layers: {len(selected)}")
print()

for item in selected:
    print(f"{item['time_ms']:.4f} ms  {item['category']}")
    print(f"layer: {item['layer_name']}")

    print("  inputs:")
    for tensor in item["inputs"]:
        print(f"    {tensor}")

    print("  outputs:")
    for tensor in item["outputs"]:
        print(f"    {tensor}")

    print()


with open(out_json, "w") as f:
    json.dump(
        {
            "source_breakdown": breakdown_json,
            "source_layers": layers_json,
            "selected_layers": selected,
        },
        f,
        indent=2,
    )

print(f"saved JSON results to {out_json}")