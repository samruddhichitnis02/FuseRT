import sys
import numpy as np
import onnxruntime as ort

CONFIGS = [  # (batch, seq_len, padded_mask)
    (1, 128, False),
    (4, 64, False),
    (8, 256, False),
    (4, 128, True),
]


def make_inputs(batch, seq, padded, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 30000, size=(batch, seq), dtype=np.int64)
    mask = np.ones((batch, seq), dtype=np.int64)
    if padded:
        mask[:, int(seq * 0.75):] = 0
    return {"input_ids": ids, "attention_mask": mask}


def run(path, feeds):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        path, sess_options=so, providers=["CPUExecutionProvider"]
    )
    return sess.run(None, feeds)


def compare(path_a, path_b, atol=1e-4):
    ok = True
    for batch, seq, padded in CONFIGS:
        feeds = make_inputs(batch, seq, padded)
        outs_a = run(path_a, feeds)
        outs_b = run(path_b, feeds)
        err = max(float(np.abs(a - b).max()) for a, b in zip(outs_a, outs_b))
        status = "PASS" if err <= atol else "FAIL"
        ok = ok and err <= atol
        print(f"batch={batch} seq={seq} padded={padded}  max_abs_err={err:.2e}  {status}")
    return ok


if __name__ == "__main__":
    good = compare(sys.argv[1], sys.argv[2])
    sys.exit(0 if good else 1)