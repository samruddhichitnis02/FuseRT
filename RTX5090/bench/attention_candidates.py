import sys
import json
import collections

profile_json = sys.argv[1]
layers_json = sys.argv[2]

with open(profile_json) as f:
    profile = json.load(f)

with open(layers_json) as f:
    layer_file = json.load(f)

layers = {
    layer["Name"]: layer
    for layer in layer_file["Layers"]
    if isinstance(layer, dict) and "Name" in layer
}


def get_layer_text(profile_layer, engine_layer):
    values = []

    values.append(str(profile_layer.get("name", "")))

    for key, value in engine_layer.items():
        if key not in ("Inputs", "Outputs"):
            values.append(str(value))

    return " ".join(values).lower()


def classify(profile_layer, engine_layer, text):
    layer_type = str(
        engine_layer.get("LayerType", profile_layer.get("type", ""))
    ).lower()

    name = str(profile_layer.get("name", "")).lower()

    if "softmax" in text:
        return "softmax"

    if layer_type == "gemm":
        return "attention gemm"

    if "tran" in name or "transpose" in text:
        return "transpose"

    if "resh" in name or "reshape" in text:
        return "reshape"

    if "cast" in name or "cast" in text:
        return "cast"

    return "other"


rows = []

for profile_layer in profile["layers"]:
    engine_layer = layers.get(profile_layer["name"], {})

    text = get_layer_text(profile_layer, engine_layer)

    # These are possible attention-related layers.
    # We include:
    # 1. layers explicitly mentioning attention/self
    # 2. softmax layers
    # 3. transpose kernels, which are often named Tran by TensorRT
    is_attention_candidate = (
        "attention/self" in text
        or "softmax" in text
        or "transpose" in text
        or "tran" in profile_layer["name"].lower()
    )

    if not is_attention_candidate:
        continue

    category = classify(profile_layer, engine_layer, text)

    rows.append(
        {
            "name": profile_layer["name"],
            "ms": float(profile_layer["ms"]),
            "category": category,
            "text": text,
        }
    )


total_profiled_ms = float(profile["sum_ms"])

by_category = collections.defaultdict(lambda: [0, 0.0])

for row in rows:
    by_category[row["category"]][0] += 1
    by_category[row["category"]][1] += row["ms"]


total_candidate_ms = sum(row["ms"] for row in rows)

print(f"profile: {profile_json}")
print(f"batch={profile['batch']}   seq={profile['seq']}")
print()

print(f"candidate layers found: {len(rows)}")
print(f"candidate total time: {total_candidate_ms:.3f} ms")
print(
    f"candidate share of profiled time: "
    f"{100 * total_candidate_ms / total_profiled_ms:.1f}%"
)
print()

print(
    f"{'category':20s}"
    f"{'count':>8s}"
    f"{'total ms':>12s}"
    f"{'share':>10s}"
)

for category, (count, ms) in sorted(
    by_category.items(),
    key=lambda item: -item[1][1],
):
    share = 100 * ms / total_profiled_ms

    print(
        f"{category:20s}"
        f"{count:>8d}"
        f"{ms:>12.3f}"
        f"{share:>9.1f}%"
    )

print()
print("candidate layers in execution order:")

for row in rows:
    print(
        f"{row['ms']:8.4f} ms  "
        f"{row['category']:18s}  "
        f"{row['name'][:100]}"
    )