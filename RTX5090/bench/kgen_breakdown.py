import sys
import json
import collections

profile_json, layers_json, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
prof = json.load(open(profile_json))
layers = {
    l["Name"]: l
    for l in json.load(open(layers_json))["Layers"]
    if isinstance(l, dict)
}


def category(text):
    if "LayerNorm" in text:
        return "layernorm (+residual add)"
    if "intermediate_act_fn" in text or "Erf" in text or "Gelu" in text:
        return "gelu"
    if "Softmax" in text:
        return "softmax"
    if "/embeddings/" in text:
        return "embeddings"
    return "other"


rows = []
first_entry = None
for l in prof["layers"]:
    if l["type"] != "kgen":
        continue
    entry = layers.get(l["name"], {})
    if first_entry is None:
        first_entry = entry
    text = l["name"] + " " + " ".join(
        str(v) for k, v in entry.items() if k not in ("Inputs", "Outputs")
    )
    rows.append((l["name"], l["ms"], category(text), text))

total_all = prof["sum_ms"]
total_kgen = sum(r[1] for r in rows)
print(f"profile: {profile_json}   batch={prof['batch']}   seq={prof['seq']}")
print(f"all layers: {total_all:.3f} ms   kgen layers: {len(rows)} = {total_kgen:.3f} ms "
      f"({100 * total_kgen / total_all:.1f}% of all)")
print()

by = collections.defaultdict(lambda: [0, 0.0])
for _, ms, cat, _ in rows:
    by[cat][0] += 1
    by[cat][1] += ms

print(f"{'category':28s}{'count':>6s}{'ms':>9s}{'% of all':>10s}{'% of kgen':>11s}")
for cat, (n, ms) in sorted(by.items(), key=lambda kv: -kv[1][1]):
    print(f"{cat:28s}{n:>6d}{ms:>9.3f}{100 * ms / total_all:>9.1f}%{100 * ms / total_kgen:>10.1f}%")

ln_ms = by["layernorm (+residual add)"][1]
print()
print(f"Upper bound: if all layernorm(+residual) layers took 0 ms, total time would fall by "
      f"{100 * ln_ms / total_all:.1f}%.")
print("A fused kernel will not reach zero, so the real gain is smaller than this.")

print()
print("fields available for a kgen layer in the layer file:",
      sorted(first_entry.keys()) if first_entry else "none")

print()
print("first 14 kgen layers in execution order:")
for name, ms, cat, text in rows[:14]:
    print(f"  {ms:7.4f} ms  {cat:26s} {name[:60]}")

other = [r for r in rows if r[2] == "other"]
if other:
    print()
    print(f"unclassified kgen layers: {len(other)}, first 8:")
    for name, ms, cat, text in other[:8]:
        print(f"  {ms:7.4f} ms  {name[:60]}  |  {text[len(name):len(name) + 90]}")

with open(out_json, "w") as f:
    json.dump(
        {"by_category": {k: {"count": v[0], "ms": v[1]} for k, v in by.items()},
         "kgen_total_ms": total_kgen, "all_ms": total_all,
         "rows": [{"name": r[0], "ms": r[1], "category": r[2]} for r in rows]},
        f, indent=1)
print()
print("saved", out_json)