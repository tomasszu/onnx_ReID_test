import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTModel:
    def __init__(self, engine_path):
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        self.input_tensor = None
        self.output_tensor = None
        self.input_memory = None
        self.output_memory = None
        self.output_buffer = None
        self.stream = cuda.Stream()

        # Initialize input/output tensor names
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_tensor = name
            else:
                self.output_tensor = name

    def _load_engine(self, engine_path):
        assert os.path.exists(engine_path)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def infer(self, input_data):
        batch_size = input_data.shape[0]
        self.context.set_input_shape(self.input_tensor, input_data.shape)

        # Allocate memory
        input_data = np.ascontiguousarray(input_data)
        input_nbytes = input_data.nbytes
        if self.input_memory is None or self.input_memory.size < input_nbytes:
            self.input_memory = cuda.mem_alloc(input_nbytes)

        out_shape = self.context.get_tensor_shape(self.output_tensor)
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_tensor))
        out_size = trt.volume(out_shape)
        if self.output_buffer is None or self.output_buffer.size < out_size:
            self.output_buffer = cuda.pagelocked_empty(out_size, out_dtype)
            self.output_memory = cuda.mem_alloc(self.output_buffer.nbytes)

        self.context.set_tensor_address(self.input_tensor, int(self.input_memory))
        self.context.set_tensor_address(self.output_tensor, int(self.output_memory))

        # Run
        cuda.memcpy_htod_async(self.input_memory, input_data, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_buffer, self.output_memory, self.stream)
        self.stream.synchronize()

        output = np.array(self.output_buffer).reshape((batch_size, -1))
        norms = np.linalg.norm(output, axis=1, keepdims=True) + 1e-12
        return output / norms
