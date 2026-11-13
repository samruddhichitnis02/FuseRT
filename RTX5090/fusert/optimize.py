import sys
import collections
import numpy as np
import onnx
import onnx_graphsurgeon as gs

SQRT2 = 1.4142135623730951
# Minimum ONNX opset each fused op needs
OPSET_NEEDED = {"LayerNormalization": 17, "Gelu": 20}


def load(path):
    return gs.import_onnx(onnx.load(path))


def save(graph, path):
    model = gs.export_onnx(graph)
    need = max([OPSET_NEEDED.get(n.op_type, 0) for n in model.graph.node] + [0])
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx") and imp.version < need:
            imp.version = need
    onnx.save(model, path)
    onnx.checker.check_model(path)


# ---------- helpers ----------
def _prod(t, op):
    """Node that produces tensor t, if it has the given op type."""
    if len(t.inputs) == 1 and t.inputs[0].op == op:
        return t.inputs[0]
    return None


def _only_consumer(t, op):
    return len(t.outputs) == 1 and t.outputs[0].op == op


def _scalar(t):
    if not isinstance(t, gs.Constant):
        return None
    v = np.asarray(t.values)
    return float(v.reshape(-1)[0]) if v.size == 1 else None


def _last_axis(node):
    axes = node.attrs.get("axes")
    return axes is not None and list(axes) == [-1]


def _other_input(node, t):
    """The input of `node` that is not tensor t."""
    return next((i for i in node.inputs if i is not t), None)


def _constant_node_value(node):
    a = node.attrs
    if "value" in a:
        v = a["value"]
        return np.asarray(v.values if isinstance(v, gs.Constant) else v)
    if "value_float" in a:
        return np.array(a["value_float"], dtype=np.float32)
    if "value_int" in a:
        return np.array(a["value_int"], dtype=np.int64)
    if "value_floats" in a:
        return np.array(a["value_floats"], dtype=np.float32)
    if "value_ints" in a:
        return np.array(a["value_ints"], dtype=np.int64)
    return None


def _report(name, fused, skipped):
    print(f"  fused {fused} {name}")
    for reason, n in skipped.items():
        print(f"  skipped {n}: {reason}")


# ---------- pass 1: constants ----------
def inline_constants(graph):
    """Turn Constant nodes into plain constant tensors (weights)."""
    for node in list(graph.nodes):
        if node.op != "Constant":
            continue
        out = node.outputs[0]
        if out in graph.outputs:
            continue
        val = _constant_node_value(node)
        if val is None:
            continue
        const = gs.Constant(name=out.name, values=val)
        for consumer in list(out.outputs):
            for i, t in enumerate(consumer.inputs):
                if t is out:
                    consumer.inputs[i] = const
        node.outputs.clear()
    graph.cleanup()


# ---------- pass 2: identity ----------
def remove_identity(graph):
    """Delete Identity nodes by connecting their input to their consumers."""
    for node in list(graph.nodes):
        if node.op != "Identity":
            continue
        inp, out = node.inputs[0], node.outputs[0]
        if out in graph.outputs:
            continue
        for consumer in list(out.outputs):
            for i, t in enumerate(consumer.inputs):
                if t is out:
                    consumer.inputs[i] = inp
        node.outputs.clear()
    graph.cleanup()


# ---------- pass 3: LayerNorm ----------
def _match_layernorm(sqrt):
    """
    ReduceMean -> Sub -> Pow(2) -> ReduceMean -> Add(eps) -> Sqrt
    -> Div -> Mul(gamma) -> Add(beta)
    """
    add_eps = _prod(sqrt.inputs[0], "Add")
    if add_eps is None:
        return None, "no Add before Sqrt"
    eps, var_t = None, None
    for t in add_eps.inputs:
        s = _scalar(t)
        if s is not None:
            eps = s
        else:
            var_t = t
    if eps is None or var_t is None:
        return None, "no scalar epsilon"

    mean2 = _prod(var_t, "ReduceMean")
    if mean2 is None:
        return None, "no ReduceMean for variance"
    if not _last_axis(mean2):
        return None, "variance axis is not last"
    pow_ = _prod(mean2.inputs[0], "Pow")
    if pow_ is None:
        return None, "no Pow"
    if _scalar(pow_.inputs[1]) != 2.0:
        return None, "Pow exponent is not 2"
    sub = _prod(pow_.inputs[0], "Sub")
    if sub is None:
        return None, "no Sub"
    x = sub.inputs[0]
    mean1 = _prod(sub.inputs[1], "ReduceMean")
    if mean1 is None or mean1.inputs[0] is not x:
        return None, "mean branch does not match"
    if not _last_axis(mean1):
        return None, "mean axis is not last"

    if not _only_consumer(sqrt.outputs[0], "Div"):
        return None, "Sqrt is not followed by Div"
    div = sqrt.outputs[0].outputs[0]
    if div.inputs[0] is not sub.outputs[0] or div.inputs[1] is not sqrt.outputs[0]:
        return None, "Div inputs do not match"
    if not _only_consumer(div.outputs[0], "Mul"):
        return None, "Div is not followed by Mul"
    mul = div.outputs[0].outputs[0]
    gamma = _other_input(mul, div.outputs[0])
    if not isinstance(gamma, gs.Constant):
        return None, "gamma is not a constant"
    if not _only_consumer(mul.outputs[0], "Add"):
        return None, "Mul is not followed by Add"
    add_b = mul.outputs[0].outputs[0]
    beta = _other_input(add_b, mul.outputs[0])
    if not isinstance(beta, gs.Constant):
        return None, "beta is not a constant"

    inner = [mean1.outputs[0], mean2.outputs[0], pow_.outputs[0], add_eps.outputs[0]]
    if any(len(t.outputs) != 1 for t in inner):
        return None, "an in-between result is used elsewhere"
    if len(sub.outputs[0].outputs) != 2:
        return None, "Sub result is used elsewhere"

    return dict(x=x, gamma=gamma, beta=beta, eps=eps, tail=add_b), None


def fuse_layernorm(graph):
    fused = 0
    skipped = collections.Counter()
    for sqrt in [n for n in graph.nodes if n.op == "Sqrt"]:
        m, reason = _match_layernorm(sqrt)
        if m is None:
            skipped[reason] += 1
            continue
        out = m["tail"].outputs[0]
        m["tail"].outputs.clear()
        graph.nodes.append(
            gs.Node(
                op="LayerNormalization",
                name=f"fused_layernorm_{fused}",
                inputs=[m["x"], m["gamma"], m["beta"]],
                outputs=[out],
                attrs={"axis": -1, "epsilon": m["eps"]},
            )
        )
        fused += 1
    graph.cleanup()
    _report("LayerNorm groups", fused, skipped)


# ---------- pass 4: GELU ----------
def _match_gelu(erf):
    """x -> Div(sqrt2) -> Erf -> Add(1) -> Mul(x, .) -> Mul(0.5)"""
    div = _prod(erf.inputs[0], "Div")
    if div is None:
        return None, "no Div before Erf"
    s = _scalar(div.inputs[1])
    if s is None or abs(s - SQRT2) > 1e-4:
        return None, "Div is not by sqrt(2)"
    x = div.inputs[0]

    if not _only_consumer(erf.outputs[0], "Add"):
        return None, "Erf is not followed by Add"
    add1 = erf.outputs[0].outputs[0]
    if _scalar(_other_input(add1, erf.outputs[0])) != 1.0:
        return None, "Add is not +1"
    if not _only_consumer(add1.outputs[0], "Mul"):
        return None, "Add is not followed by Mul"
    mul1 = add1.outputs[0].outputs[0]
    if not any(t is x for t in mul1.inputs):
        return None, "first Mul does not use x"
    if not _only_consumer(mul1.outputs[0], "Mul"):
        return None, "first Mul is not followed by Mul"
    mul2 = mul1.outputs[0].outputs[0]
    if _scalar(_other_input(mul2, mul1.outputs[0])) != 0.5:
        return None, "last Mul is not * 0.5"

    if len(x.outputs) != 2:
        return None, "x is used elsewhere"
    inner = [div.outputs[0], erf.outputs[0], add1.outputs[0], mul1.outputs[0]]
    if any(len(t.outputs) != 1 for t in inner):
        return None, "an in-between result is used elsewhere"
    return dict(x=x, tail=mul2), None


def fuse_gelu(graph):
    fused = 0
    skipped = collections.Counter()
    for erf in [n for n in graph.nodes if n.op == "Erf"]:
        m, reason = _match_gelu(erf)
        if m is None:
            skipped[reason] += 1
            continue
        out = m["tail"].outputs[0]
        m["tail"].outputs.clear()
        graph.nodes.append(
            gs.Node(
                op="Gelu",
                name=f"fused_gelu_{fused}",
                inputs=[m["x"]],
                outputs=[out],
                attrs={"approximate": "none"},
            )
        )
        fused += 1
    graph.cleanup()
    _report("GELU groups", fused, skipped)


# ---------- pass 5: QKV merge ----------
def _match_qkv(h):
    """Tensor h read by exactly 3 MatMuls, each followed by a bias Add."""
    mms = [c for c in h.outputs if c.op == "MatMul"]
    if len(mms) != 3:
        return None, "not exactly 3 MatMuls"
    parts = []
    for mm in mms:
        if mm.inputs[0] is not h:
            return None, "h is not the left MatMul input"
        w = mm.inputs[1]
        if not isinstance(w, gs.Constant) or w.values.ndim != 2:
            return None, "weight is not a 2D constant"
        if not _only_consumer(mm.outputs[0], "Add"):
            return None, "MatMul is not followed by Add"
        add = mm.outputs[0].outputs[0]
        bias = _other_input(add, mm.outputs[0])
        if not isinstance(bias, gs.Constant) or bias.values.ndim != 1:
            return None, "bias is not a 1D constant"
        if bias.values.shape[0] != w.values.shape[1]:
            return None, "bias size does not match weight"
        parts.append((mm, w, add, bias))
    if len({p[1].values.shape[0] for p in parts}) != 1:
        return None, "weights have different input sizes"
    return parts, None


def merge_qkv(graph):
    merged = 0
    skipped = collections.Counter()
    for node in list(graph.nodes):
        for h in list(node.outputs):
            if sum(1 for c in h.outputs if c.op == "MatMul") != 3:
                continue
            parts, reason = _match_qkv(h)
            if parts is None:
                skipped[reason] += 1
                continue

            Ws = [p[1].values for p in parts]
            bs = [p[3].values for p in parts]
            W = np.concatenate(Ws, axis=1)
            b = np.concatenate(bs, axis=0)
            sizes = np.array([w.shape[1] for w in Ws], dtype=np.int64)

            mm_out = gs.Variable(f"qkv_matmul_out_{merged}", dtype=W.dtype)
            add_out = gs.Variable(f"qkv_add_out_{merged}", dtype=W.dtype)
            split_outs = [p[2].outputs[0] for p in parts]  # keep original tensors
            for p in parts:
                p[2].outputs.clear()

            graph.nodes.append(gs.Node(
                op="MatMul", name=f"qkv_matmul_{merged}",
                inputs=[h, gs.Constant(f"qkv_W_{merged}", W)], outputs=[mm_out]))
            graph.nodes.append(gs.Node(
                op="Add", name=f"qkv_bias_{merged}",
                inputs=[mm_out, gs.Constant(f"qkv_b_{merged}", b)], outputs=[add_out]))
            graph.nodes.append(gs.Node(
                op="Split", name=f"qkv_split_{merged}",
                inputs=[add_out, gs.Constant(f"qkv_sizes_{merged}", sizes)],
                outputs=split_outs, attrs={"axis": -1}))
            merged += 1
    graph.cleanup()
    _report("QKV groups", merged, skipped)


# ---------- cleanup ----------
def dead_code_elimination(graph):
    graph.cleanup()
    graph.toposort()


PASSES = [inline_constants, remove_identity, fuse_layernorm,
          fuse_gelu, merge_qkv, dead_code_elimination]


def optimize(graph):
    print(f"start: {len(graph.nodes)} nodes")
    for p in PASSES:
        before = len(graph.nodes)
        p(graph)
        print(f"{p.__name__:28s}{before} -> {len(graph.nodes)}")
    return graph


if __name__ == "__main__":
    g = load(sys.argv[1])
    g = optimize(g)
    save(g, sys.argv[2])
    print("saved", sys.argv[2])