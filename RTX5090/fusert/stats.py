import sys
import collections
import onnx

m = onnx.load(sys.argv[1])
g = m.graph
counts = collections.Counter(n.op_type for n in g.node)

print("total nodes:", len(g.node))
print("weights (initializers):", len(g.initializer))
print("inputs:", [i.name for i in g.input])
print("outputs:", [o.name for o in g.output])
print()
for op, n in counts.most_common():
    print(f"{op:28s}{n}")