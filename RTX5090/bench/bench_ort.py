import time
import numpy as np
import onnxruntime as ort

ort.preload_dlls()

MODELS = [("original", "models/bert.onnx"), ("optimized", "models/bert_opt.onnx")]
LEVELS = [
    ("ORT_off", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
    ("ORT_default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
]
BATCHES = [1, 8, 32]
SEQ = 128


def bench(path, level, batch, seq=SEQ, iters=200, warmup=50):
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    sess = ort.InferenceSession(path, so, providers=["CUDAExecutionProvider"])
    if sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA provider not active: {sess.get_providers()}")

    rng = np.random.default_rng(0)
    ids = ort.OrtValue.ortvalue_from_numpy(
        rng.integers(0, 30000, (batch, seq), dtype=np.int64), "cuda", 0)
    mask = ort.OrtValue.ortvalue_from_numpy(
        np.ones((batch, seq), dtype=np.int64), "cuda", 0)
    out0 = ort.OrtValue.ortvalue_from_shape_and_type(
        (batch, seq, 768), np.float32, "cuda", 0)
    out1 = ort.OrtValue.ortvalue_from_shape_and_type(
        (batch, 768), np.float32, "cuda", 0)

    io = sess.io_binding()
    io.bind_ortvalue_input("input_ids", ids)
    io.bind_ortvalue_input("attention_mask", mask)
    io.bind_ortvalue_output("last_hidden_state", out0)
    io.bind_ortvalue_output("pooler_output", out1)

    for _ in range(warmup):
        sess.run_with_iobinding(io)
    io.synchronize_outputs()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run_with_iobinding(io)
        io.synchronize_outputs()
        times.append((time.perf_counter() - t0) * 1000)
    return np.percentile(times, [50, 95, 99])


print(f"ORT device: {ort.get_device()}   seq={SEQ}   FP32   times in ms")
print(f"{'model':10s}{'ORT level':13s}{'batch':>6s}{'p50':>9s}{'p95':>9s}{'p99':>9s}{'samples/s':>11s}")
for mname, path in MODELS:
    for lname, level in LEVELS:
        for b in BATCHES:
            p50, p95, p99 = bench(path, level, b)
            print(f"{mname:10s}{lname:13s}{b:>6d}{p50:>9.2f}{p95:>9.2f}{p99:>9.2f}{b / p50 * 1000:>11.1f}")