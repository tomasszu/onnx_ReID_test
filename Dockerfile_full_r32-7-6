FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.9-py3

# Use the official NVIDIA L4T PyTorch image as a base
# This image is optimized for Jetson devices and includes PyTorch pre-installed

# Set environment variables
# These variables help in configuring the environment for non-interactive installations
# and to ensure that Python does not write bytecode files or buffer output
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Jo te nav kkads public key prieks taa public repo
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 16FAAD7AF99A65E2

# Install system dependencies
# These packages are necessary for building and running the application
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    libx11-6 \
    build-essential \
    screen \
    libgl1 \
    libglib2.0-0 \
    libfreetype6-dev \
    pkg-config \
    python3-opencv \
    python3-pip \
    nano \
    protobuf-compiler \
    libprotoc-dev

# Optional symlink
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy the requirements file into the container
# This file contains the Python package dependencies for the application
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

# Install specific versions of packages
# These versions are specified to ensure compatibility with the application
RUN python3 -m pip install protobuf==3.19.6
RUN wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
RUN python3 -m pip install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl

RUN python3 -m pip install onnx==1.11.0

COPY . /app

CMD ["python3", "main.py"]
