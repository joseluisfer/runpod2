# -------------------------------------------------
# Base PyTorch + CUDA
# -------------------------------------------------
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -------------------------------------------------
# Dependencias del sistema
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# Instalar dependencias Python
# -------------------------------------------------
RUN pip install --upgrade pip

RUN pip install \
    runpod \
    ultralytics \
    pillow \
    requests \
    ftfy \
    regex \
    tqdm

# CLIP necesario para YOLO World
RUN pip install git+https://github.com/openai/CLIP.git

# FORZAR instalación de numpy al final
RUN pip install --force-reinstall numpy==1.26.4

# -------------------------------------------------
# Descargar modelo durante build
# -------------------------------------------------
RUN python -c "from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')"

# -------------------------------------------------
# Copiar handler
# -------------------------------------------------
COPY handler.py .

# -------------------------------------------------
# Start
# -------------------------------------------------
CMD ["python","-u","handler.py"]
