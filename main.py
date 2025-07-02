import os
import time
import glob
import cv2
import numpy as np
import psutil
import onnxruntime as ort
import shutil
import subprocess
import re


# ImageNet normalization values
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(img, target_size=(224, 224)):
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return img

def yield_image_batches(image_paths, batch_size):
    batch = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue  # skip unreadable
        batch.append(preprocess_image(img))
        if len(batch) == batch_size:
            yield np.stack(batch), batch
            batch = []
    if batch:
        yield np.stack(batch), batch

def run_inference(session, inputs):
    ort_inputs = {session.get_inputs()[0].name: inputs}
    feats = session.run(None, ort_inputs)[0]
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    return feats / norms

def print_mem_usage(label="MEM"):
    # CPU RAM usage for current Python process
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    gpu_mem = "N/A"

    # Check for Jetson tegrastats
    if shutil.which("tegrastats"):
        try:
            output = subprocess.check_output(['tegrastats', '--interval', '100', '--count', '1'], stderr=subprocess.DEVNULL)
            line = output.decode().strip().split('\n')[0]
            match = re.search(r'RAM (\d+)/\d+MB', line)
            if match:
                gpu_mem = f"{int(match.group(1))} MB (Jetson)"
        except Exception as e:
            gpu_mem = "Error"
    
    # Otherwise try nvidia-smi (workstation with discrete GPU)
    elif shutil.which("nvidia-smi"):
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                stderr=subprocess.DEVNULL
            )
            gpu_mem_raw = output.decode().strip().split('\n')
            gpu_mem = f"{gpu_mem_raw[0]} MB (CUDA GPU)"
        except Exception as e:
            gpu_mem = "Error"

    print(f"[{label}] CPU RAM: {cpu_mem:.2f} MB | GPU RAM: {gpu_mem}")

def main():
    model_path = "model/net_39_opt.onnx"
    image_folder = "gallery_images"
    batch_size = 1  # simulate streaming
    use_cuda = True

    print_mem_usage("Before model load")
    providers = ['CUDAExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    print_mem_usage("After model load")

    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")) +
                         glob.glob(os.path.join(image_folder, "*.png")))
    print(f"Found {len(image_paths)} images")

    total_time = 0.0
    total_vectors = 0
    total_feature_norm = 0.0  # optional: for validation

    for batch_input_np, _ in yield_image_batches(image_paths, batch_size):
        batch_input_np = np.ascontiguousarray(batch_input_np)
        start = time.time()
        feats = run_inference(session, batch_input_np)
        end = time.time()

        total_time += (end - start)
        total_vectors += feats.shape[0]
        total_feature_norm += np.linalg.norm(feats)

    print(f"Extracted {total_vectors} vectors")
    print(f"Total inference time: {total_time:.2f}s | Avg per image: {total_time / total_vectors:.4f}s")
    print(f"Average per image: {total_time / total_vectors:.4f}s | {(total_time / total_vectors) * 1000:.2f} ms")
    print_mem_usage("After inference")

if __name__ == "__main__":
    main()
