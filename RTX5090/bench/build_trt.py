import sys
import json
import time
import tensorrt as trt

onnx_path, plan_path, layers_path = sys.argv[1], sys.argv[2], sys.argv[3]

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

if not parser.parse_from_file(onnx_path):
    for i in range(parser.num_errors):
        print("PARSE ERROR:", parser.get_error(i))
    sys.exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

profile = builder.create_optimization_profile()
for i in range(network.num_inputs):
    name = network.get_input(i).name
    profile.set_shape(name, (1, 128), (8, 128), (32, 128))
config.add_optimization_profile(profile)

t0 = time.time()
blob = builder.build_serialized_network(network, config)
if blob is None:
    print("BUILD FAILED")
    sys.exit(1)
print(f"build time {time.time() - t0:.1f}s, engine size {blob.nbytes / 1e6:.1f} MB")
with open(plan_path, "wb") as f:
    f.write(bytes(blob))

engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
info = engine.create_engine_inspector().get_engine_information(
    trt.LayerInformationFormat.JSON
)
with open(layers_path, "w") as f:
    f.write(info)
print("layers in engine:", len(json.loads(info)["Layers"]))