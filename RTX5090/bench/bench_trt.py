import time
import numpy as np
import torch
import tensorrt as trt

ENGINES = [("original", "models/bert_fp32.plan"), ("optimized", "models/bert_opt_fp32.plan")]
BATCHES = [1, 8, 32]
SEQ = 128
TORCH_DTYPE = {
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.FLOAT: torch.float32,
}


def bench(plan, batch, iters=200, warmup=50):
    logger = trt.Logger(trt.Logger.WARNING)
    engine = trt.Runtime(logger).deserialize_cuda_engine(open(plan, "rb").read())
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
        for _ in range(warmup):
            ctx.execute_async_v3(h)
        stream.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            ctx.execute_async_v3(h)
            stream.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return np.percentile(times, [50, 95, 99])


print(f"TensorRT {trt.__version__}   seq={SEQ}   FP32   times in ms")
print(f"{'engine':10s}{'batch':>6s}{'p50':>9s}{'p95':>9s}{'p99':>9s}{'samples/s':>11s}")
for name, plan in ENGINES:
    for b in BATCHES:
        p50, p95, p99 = bench(plan, b)
        print(f"{name:10s}{b:>6d}{p50:>9.2f}{p95:>9.2f}{p99:>9.2f}{b / p50 * 1000:>11.1f}")