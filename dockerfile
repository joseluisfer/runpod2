FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalamos las librerías
RUN pip install --no-cache-dir runpod ultralytics requests pillow

# 🔥 ESTA ES LA LÍNEA MÁGICA: Descarga el modelo World v2-X automáticamente
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')"

COPY handler.py /app/


CMD ["python", "-u", "/app/handler.py"]
