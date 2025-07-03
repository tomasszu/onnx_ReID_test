FROM dustynv/onnxruntime:r32.7.1

# Install Python packages
RUN apt-get update && apt-get install -y python3-pip python3-opencv
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy your code
COPY . /app/
WORKDIR /app/

CMD ["python3", "main.py"]
