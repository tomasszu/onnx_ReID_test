FROM nvcr.io/nvidia/l4t-tensorrt:r10.3.0-devel

# Nano nav iekļauts (apt update && apt install nano -y)

WORKDIR /app

# pip install pycuda

# Optional: install ONNX Runtime (GPU support via TensorRT)
RUN pip install --no-cache-dir onnxruntime-gpu==1.17.1 numpy opencv-python

# Add your model + inference script
COPY model_fp16.engine .
COPY reid_infer.py .

# Run inference
CMD ["python3", "reid_infer.py"]
