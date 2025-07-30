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

        # Allocate host and device buffers
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for tensor in tensor_names:
            size = trt.volume(context.get_tensor_shape(tensor))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor))

            if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                context.set_input_shape(tensor, input_data.shape)
                input_buffer = np.ascontiguousarray(input_data)
                input_memory = cuda.mem_alloc(input_buffer.nbytes)
                context.set_tensor_address(tensor, int(input_memory))
                print(f"Input tensor {tensor} allocated with shape {input_data.shape} and dtype {input_buffer.dtype}")
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                context.set_tensor_address(tensor, int(output_memory))
                print(f"Output tensor {tensor} allocated with shape {context.get_tensor_shape(tensor)} and dtype {output_buffer.dtype}")

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
result = infer("net_39_opt_docker_built.engine", dummy_input)
print("Inference output:", result)
