FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias Python
RUN pip install --no-cache-dir \
    numpy \
    runpod \
    ultralytics \
    requests \
    pillow \
    git+https://github.com/openai/CLIP.git

# Descargar el modelo durante el build (reduce cold start)
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')"

# Copiar handler
COPY handler.py /app/

# Ejecutar worker serverless
CMD ["python", "-u", "/app/handler.py"]



