FROM nvcr.io/nvidia/l4t-tensorrt:r10.3.0-devel

# Nano nav iekļauts (apt update && apt install nano -y)

#vajag cv2
# pip install pycuda, psutil

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Install Python packages
RUN apt-get update && apt-get install -y python3-opencv
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your code
COPY . /app/
WORKDIR /app/

# Run inference
CMD ["python", "main.py"]
