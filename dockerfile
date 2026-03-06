FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 🔥 ACTUALIZADO: Añadimos numpy explícitamente
RUN pip install --no-cache-dir numpy runpod ultralytics requests pillow

# Descarga del modelo
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')"

COPY handler.py /app/

CMD ["python", "-u", "/app/handler.py"]
