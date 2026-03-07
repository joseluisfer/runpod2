# ===============================================
# Dockerfile para YOLOv8x-WORLD en Runpod
# ===============================================

FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Instalación de dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalación de dependencias Python
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    runpod \
    ultralytics>=8.1.0 \
    pillow \
    requests \
    ftfy \
    regex \
    tqdm \
    clip

# Copiar handler
COPY handler.py /app/

# Pre-descarga del modelo (warm start)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')"

# Puerto por defecto y comando
CMD ["python", "-u", "/app/handler.py"]
