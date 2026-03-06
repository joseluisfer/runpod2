FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    runpod \
    ultralytics \
    pillow \
    requests

# descargar modelo en build
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
