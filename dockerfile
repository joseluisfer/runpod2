FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# instalar python y dependencias sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# actualizar pip
RUN pip3 install --upgrade pip

# instalar pytorch GPU
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# numpy compatible
RUN pip3 install --no-cache-dir numpy==1.26.4

# dependencias inference
RUN pip3 install --no-cache-dir \
    runpod \
    ultralytics \
    pillow \
    requests \
    opencv-python-headless \
    git+https://github.com/openai/CLIP.git

# descargar modelo durante build (reduce cold start)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')"

# copiar handler
COPY handler.py /app/

# start worker
CMD ["python3", "-u", "handler.py"]
