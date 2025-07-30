import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Required to initialize CUDA driver

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load the TensorRT engine
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())



def infer(engine_path, input_data):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    stream = cuda.Stream()

    for name in engine:
        print(f"Tensor: {name}, Mode: {engine.get_tensor_mode(name)}, Shape: {engine.get_tensor_shape(name)}")

    # Allocate input
    input_name = [name for name in engine if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT][0]
    input_shape = engine.get_tensor_shape(input_name)
    input_size = trt.volume(input_shape)
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    h_input = cuda.pagelocked_empty(input_size, input_dtype)
    d_input = cuda.mem_alloc(h_input.nbytes)

    np.copyto(h_input, input_data.ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)

    context.set_tensor_address(input_name, int(d_input))

    # Allocate output
    output_name = [name for name in engine if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT][0]
    output_shape = engine.get_tensor_shape(output_name)
    output_size = trt.volume(output_shape)
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    h_output = cuda.pagelocked_empty(output_size, output_dtype)
    d_output = cuda.mem_alloc(h_output.nbytes)

    context.set_tensor_address(output_name, int(d_output))

    # Run inference
    context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    return h_output


# Example input
input_shape = (1, 3, 224, 224)  # change this to match your engine
np.random.seed(42)
dummy_input = np.random.rand(*input_shape).astype(np.float32)
result = infer("net_39_opt_docker_built.engine", dummy_input)
print("Inference output:", result)
