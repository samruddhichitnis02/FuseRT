import sys
import json
import collections


def load(path):
    layers = json.load(open(path))["Layers"]
    out = []
    for l in layers:
        if isinstance(l, dict):
            out.append((l.get("LayerType", "?"), l.get("Name", "?")))
        else:
            out.append(("?", str(l)))
    return out


a = load(sys.argv[1])
b = load(sys.argv[2])
ca = collections.Counter(t for t, _ in a)
cb = collections.Counter(t for t, _ in b)

print(f"{'layer type':32s}{'original':>10s}{'optimized':>11s}")
for t in sorted(set(ca) | set(cb), key=lambda t: -(ca[t] + cb[t])):
    print(f"{t:32s}{ca[t]:>10d}{cb[t]:>11d}")
print()
print("total layers:", len(a), len(b))
names_a = sorted(n for _, n in a)
names_b = sorted(n for _, n in b)
print("identical layer names:", names_a == names_b)

only_a = sorted(set(names_a) - set(names_b))
only_b = sorted(set(names_b) - set(names_a))
print("only in original (first 8):")
for n in only_a[:8]:
    print("  ", n[:110])
print("only in optimized (first 8):")
for n in only_b[:8]:
    print("  ", n[:110])