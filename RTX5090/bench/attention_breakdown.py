import sys
import json
import collections

profile_json = sys.argv[1]
layers_json = sys.argv[2]
out_json = sys.argv[3]

with open(profile_json) as f:
    profile = json.load(f)

with open(layers_json) as f:
    layer_file = json.load(f)

layers = {
    layer["Name"]: layer
    for layer in layer_file["Layers"]
    if isinstance(layer, dict) and "Name" in layer
}


def make_text(profile_layer, engine_layer):
    parts = [str(profile_layer.get("name", ""))]

    for key, value in engine_layer.items():
        if key not in ("Inputs", "Outputs"):
            parts.append(str(value))

    return " ".join(parts).lower()


def classify(profile_layer, engine_layer, text):
    name = str(profile_layer.get("name", "")).lower()
    layer_type = str(
        engine_layer.get("LayerType", profile_layer.get("type", ""))
    ).lower()

    # Softmax inside attention
    if "softmax" in text:
        return "softmax"

    # Final attention output projection.
    if "/attention/output/dense/matmul" in text:
        return "output_projection_gemm"

    # Combined Q/K/V projection created by FuseRT merge_qkv.
    if (
        "/attention/self/value/matmul+" in text
        and "/attention/self/key/matmul" in text
    ):
        return "qkv_projection_gemm"

    # Attention scores multiplied by V.
    if "/attention/self/matmul_1" in text:
        return "scores_times_v_gemm"

    # Q multiplied by K transpose.
    if "/attention/self/matmul" in text:
        return "q_times_k_transpose_gemm"

    # TensorRT-generated transpose kernels.
    if layer_type == "kgen" and (
        "tran" in name or "transpose" in text
    ):
        return "attention_transpose_candidate"

    return None


rows = []

for profile_layer in profile["layers"]:
    engine_layer = layers.get(profile_layer["name"], {})
    text = make_text(profile_layer, engine_layer)

    category = classify(profile_layer, engine_layer, text)

    if category is None:
        continue

    rows.append(
        {
            "name": profile_layer["name"],
            "ms": float(profile_layer["ms"]),
            "category": category,
        }
    )


total_profiled_ms = float(profile["sum_ms"])

by_category = collections.defaultdict(lambda: [0, 0.0])

for row in rows:
    by_category[row["category"]][0] += 1
    by_category[row["category"]][1] += row["ms"]


def category_time(category):
    return by_category[category][1]


qk_ms = category_time("q_times_k_transpose_gemm")
softmax_ms = category_time("softmax")
scores_v_ms = category_time("scores_times_v_gemm")
transpose_ms = category_time("attention_transpose_candidate")

attention_core_ms = qk_ms + softmax_ms + scores_v_ms
attention_with_transposes_ms = attention_core_ms + transpose_ms


print(f"profile: {profile_json}")
print(f"batch={profile['batch']}   seq={profile['seq']}")
print(f"total profiled time: {total_profiled_ms:.3f} ms")
print()

print("detailed category breakdown")
print(
    f"{'category':35s}"
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
        f"{category:35s}"
        f"{count:>8d}"
        f"{ms:>12.3f}"
        f"{share:>9.1f}%"
    )

print()
print("possible plugin target")
print(
    f"Q x K^T + softmax + scores x V: "
    f"{attention_core_ms:.3f} ms "
    f"({100 * attention_core_ms / total_profiled_ms:.1f}% of profiled time)"
)

print(
    f"That group + transpose candidates: "
    f"{attention_with_transposes_ms:.3f} ms "
    f"({100 * attention_with_transposes_ms / total_profiled_ms:.1f}% "
    f"of profiled time)"
)

print()
print("selected layers:")

for row in rows:
    print(
        f"{row['ms']:8.4f} ms  "
        f"{row['category']:35s}  "
        f"{row['name'][:110]}"
    )


result = {
    "profile": profile_json,
    "layers": layers_json,
    "batch": profile["batch"],
    "seq": profile["seq"],
    "total_profiled_ms": total_profiled_ms,
    "by_category": {
        category: {
            "count": count,
            "ms": ms,
            "share_of_profiled_time": 100 * ms / total_profiled_ms,
        }
        for category, (count, ms) in by_category.items()
    },
    "possible_plugin_target": {
        "attention_core_ms": attention_core_ms,
        "attention_core_share": 100 * attention_core_ms / total_profiled_ms,
        "attention_with_transposes_ms": attention_with_transposes_ms,
        "attention_with_transposes_share": (
            100 * attention_with_transposes_ms / total_profiled_ms
        ),
    },
    "rows": rows,
}

with open(out_json, "w") as f:
    json.dump(result, f, indent=2)

print()
print(f"saved JSON results to {out_json}")