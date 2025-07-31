import tensorrt as trt
import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Required to initialize CUDA driver

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load the TensorRT engine
def load_engine(engine_path):
    assert os.path.exists(engine_path)
    print(f"Loading TensorRT engine from {engine_path}")
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())



def infer(engine_path, input_data):
    engine = load_engine(engine_path)
    with engine.create_execution_context() as context:

        # Allocate input memory
        input_tensor_name = 'input'  # Assuming the input tensor is named 'input'
        context.set_input_shape(input_tensor_name, input_data.shape)

        input_buffer = np.ascontiguousarray(input_data)
        input_memory = cuda.mem_alloc(input_buffer.nbytes)

        context.set_tensor_address(input_tensor_name, int(input_memory))
        
        # Allocate output memory
        output_tensor_name = 'output'  # Assuming the output tensor is named 'output'

        out_size = trt.volume(context.get_tensor_shape(output_tensor_name))
        out_dtype = trt.nptype(engine.get_tensor_dtype(output_tensor_name))

        output_buffer = cuda.pagelocked_empty(out_size, out_dtype)
        output_memory = cuda.mem_alloc(output_buffer.nbytes)

        context.set_tensor_address(output_tensor_name, int(output_memory))

        stream = cuda.Stream()

        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v3(stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)

        # Synchronize the stream
        stream.synchronize()

        output_d32 = np.array(output_buffer, dtype=np.float32)

    

    return output_d32



# Example input
input_shape = (1, 3, 224, 224)  # change this to match your engine
np.random.seed(42)
dummy_input = np.random.rand(*input_shape).astype(np.float32)
result = infer("model/net_39_opt_docker_built.engine", dummy_input)
print("Inference output:", result)
