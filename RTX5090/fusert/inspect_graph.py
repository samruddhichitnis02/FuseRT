import sys
import onnx_graphsurgeon as gs
from fusert.optimize import load, inline_constants


def desc(t):
    if isinstance(t, gs.Constant):
        v = t.values
        if v.size == 1:
            return f"const={v.reshape(-1)[0]}"
        return f"const{list(v.shape)}"
    return "tensor"


def up(t, depth, pad="    "):
    if depth == 0 or not t.inputs:
        return
    n = t.inputs[0]
    print(f"{pad}{n.op} <- {[desc(i) for i in n.inputs]}")
    for i in n.inputs:
        up(i, depth - 1, pad + "   ")


def down(t, depth, pad="    "):
    if depth == 0:
        return
    for n in t.outputs:
        print(f"{pad}{n.op} <- {[desc(i) for i in n.inputs]}")
        for o in n.outputs:
            down(o, depth - 1, pad + "   ")


g = load(sys.argv[1])
inline_constants(g)

print("=== FIRST GELU (around the first Erf) ===")
erf = next(n for n in g.nodes if n.op == "Erf")
print("Erf input comes from:")
up(erf.inputs[0], 2)
print("Erf output goes to:")
down(erf.outputs[0], 3)

print()
print("=== FIRST QKV (tensor read by 3 MatMuls) ===")
for n in g.nodes:
    for out in n.outputs:
        mms = [c for c in out.outputs if c.op == "MatMul"]
        if len(mms) >= 3:
            print(f"tensor from {n.op} feeds {len(mms)} MatMuls")
            for m in mms:
                print("  MatMul <-", [desc(i) for i in m.inputs])
                down(m.outputs[0], 3, "      ")
            raise SystemExit