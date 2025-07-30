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
import gc



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

# Read tegrastats once to get GPU memory usage on Jetson devices
# This function reads the output of the tegrastats command
# and extracts the used GPU memory.
def read_tegrastats_once():
    try:
        #Using Popen + .readline() ensures we wait for first output, then terminate immediately.
        proc = subprocess.Popen(
            ["tegrastats"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        # Read one line from the output and kill the process
        output = proc.stdout.readline().decode()
        proc.terminate()
        match = re.search(r'RAM (\d+)/\d+MB', output)
        if match:
            return f"{int(match.group(1))} MB (Jetson tegrastats)"
        return "N/A"
    except Exception as e:
        return f"Error: {str(e)}"


def print_mem_usage(label="MEM"):
    # CPU RAM usage for current Python process
    cpu_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    gpu_mem = "N/A"

    # Check for Jetson tegrastats
    if shutil.which("tegrastats"):
        gpu_mem = read_tegrastats_once()
    
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
    # Jetson fallback: read from mem_info if available
    elif os.path.exists("/proc/driver/nvidia/gpus/0/mem_info"):
        try:
            with open("/proc/driver/nvidia/gpus/0/mem_info") as f:
                lines = f.readlines()
            used_line = [line for line in lines if "Used" in line][0]
            used_kb = int(re.findall(r'\d+', used_line)[0])
            gpu_mem = f"{used_kb // 1024} MB (Jetson /proc)"
        except Exception:
            gpu_mem = "Error"

    print(f"[{label}] CPU RAM: {cpu_mem:.2f} MB | GPU RAM: {gpu_mem}")

def main():
    model_path = "model/net_39_opt.onnx"
    image_folder = "gallery_images"
    batch_size = 1  # simulate streaming
    use_cuda = True

    print_mem_usage("Before model load")
    providers = ['CUDAExecutionProvider'] if use_cuda else ['CPUExecutionProvider']

    #turn off thread affinity warnings on Jetson devices
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1  # Disable thread pinning

    session = ort.InferenceSession(model_path, sess_options, providers=providers)

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

    # Explicit cleanup to avoid free() crash (nav veel izmeeginaats)
    del session
    gc.collect()
    print("Memory cleanup done")
    print_mem_usage("After cleanup")


if __name__ == "__main__":
    main()
