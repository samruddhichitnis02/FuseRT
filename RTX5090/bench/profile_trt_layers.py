import sys
import json
import collections
import torch
import tensorrt as trt

plan_path, layers_json, batch, out_json = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
SEQ = 128
WARMUP, ITERS = 30, 100

TORCH_DTYPE = {
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.FLOAT: torch.float32,
}


class LayerProfiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.total_ms = collections.OrderedDict()

    def report_layer_time(self, layer_name, ms):
        self.total_ms[layer_name] = self.total_ms.get(layer_name, 0.0) + ms


logger = trt.Logger(trt.Logger.WARNING)
engine = trt.Runtime(logger).deserialize_cuda_engine(open(plan_path, "rb").read())
ctx = engine.create_execution_context()
stream = torch.cuda.Stream()

bufs = {}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        ctx.set_input_shape(name, (batch, SEQ))
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = tuple(ctx.get_tensor_shape(name))
    dt = TORCH_DTYPE[engine.get_tensor_dtype(name)]
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        t = torch.randint(0, 30000, shape, device="cuda").to(dt)
        if name == "attention_mask":
            t = torch.ones(shape, dtype=dt, device="cuda")
    else:
        t = torch.empty(shape, dtype=dt, device="cuda")
    bufs[name] = t
    ctx.set_tensor_address(name, t.data_ptr())

h = stream.cuda_stream
with torch.cuda.stream(stream):
    for _ in range(WARMUP):
        ctx.execute_async_v3(h)
    stream.synchronize()
    prof = LayerProfiler()
    ctx.profiler = prof
    for _ in range(ITERS):
        ctx.execute_async_v3(h)
        stream.synchronize()

types = {}
for layer in json.load(open(layers_json))["Layers"]:
    types[layer["Name"]] = layer.get("LayerType", "?")

rows = [(n, ms / ITERS, types.get(n, "?")) for n, ms in prof.total_ms.items()]
total = sum(r[1] for r in rows)
matched = sum(1 for r in rows if r[2] != "?")

print(f"engine: {plan_path}   batch={batch}   seq={SEQ}   iterations={ITERS}")
print(f"layers profiled: {len(rows)}   names matched to layer file: {matched}")
print(f"sum of layer times: {total:.3f} ms")
print()

by_type = collections.defaultdict(lambda: [0, 0.0])
for _, ms, t in rows:
    by_type[t][0] += 1
    by_type[t][1] += ms
print(f"{'layer type':16s}{'count':>7s}{'total ms':>11s}{'share':>9s}")
for t, (n, ms) in sorted(by_type.items(), key=lambda kv: -kv[1][1]):
    print(f"{t:16s}{n:>7d}{ms:>11.3f}{100 * ms / total:>8.1f}%")

print()
print("slowest 15 layers:")
for n, ms, t in sorted(rows, key=lambda r: -r[1])[:15]:
    print(f"{ms:8.4f} ms  {100 * ms / total:5.1f}%  {t:8s} {n[:90]}")

out = {
    "engine": plan_path, "batch": batch, "seq": SEQ, "iterations": ITERS,
    "sum_ms": total,
    "layers": [{"name": n, "ms": ms, "type": t, "pct": 100 * ms / total} for n, ms, t in rows],
}
with open(out_json, "w") as f:
    json.dump(out, f, indent=1)
print()
print("saved", out_json)