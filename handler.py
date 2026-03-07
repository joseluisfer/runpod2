import runpod
import torch
import os
import requests
import base64
import time
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

# ==========================================================
# 🔥 CONFIGURACIÓN GLOBAL ULTRA OPTIMIZADA
# ==========================================================

print("\n🚀 INICIANDO YOLOv8x-WORLD v2 (MODO MAX PERFORMANCE)\n")

if not torch.cuda.is_available():
    raise RuntimeError("❌ ESTE ENDPOINT REQUIERE GPU")

DEVICE = 0
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

print(f"✅ GPU: {torch.cuda.get_device_name(DEVICE)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(DEVICE).total_memory / 1e9:.2f} GB")

# ==========================================================
# 🔥 CARGA DEL MODELO (WARM START REAL)
# ==========================================================

print("\n📥 Cargando yolov8x-worldv2.pt en GPU...")

model = YOLO("yolov8x-worldv2.pt")
model.to(f"cuda:{DEVICE}")
model.fuse()
model.model.half()  # FP16

print("✅ Modelo cargado en GPU con FP16")

# ==========================================================
# 🔥 PRECALENTAMIENTO REAL (ELIMINA LATENCIA INICIAL)
# ==========================================================

print("🔥 Ejecutando warmup...")

dummy = torch.zeros(1, 3, 640, 640).to(f"cuda:{DEVICE}").half()

with torch.inference_mode():
    _ = model.predict(
        source=dummy,
        imgsz=640,
        device=DEVICE,
        verbose=False
    )

torch.cuda.synchronize()
print("✅ Warmup completado\n")

# ==========================================================
# 📦 UTILIDADES
# ==========================================================

def download_image(image_input):
    """Descarga o decodifica imagen desde URL o base64"""
    try:
        if image_input.startswith(("http://", "https://")):
            response = requests.get(image_input, timeout=15)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            if "base64," in image_input:
                image_input = image_input.split("base64,")[1]
            image_data = base64.b64decode(image_input)
            return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise Exception(f"Error procesando la imagen: {e}")

# ==========================================================
# ⚡ HANDLER OPTIMIZADO
# ==========================================================

def handler(job):
    global model
    start_time = time.time()

    job_input = job.get("input", {})

    if "image" not in job_input:
        return {"error": "No image provided"}

    # Descargar imagen
    image = download_image(job_input["image"])
    temp_path = f"/tmp/{job.get('id', 'temp')}.jpg"
    image.save(temp_path, "JPEG", quality=95)

    # Clases
    custom_classes = job_input.get("classes", ["person"])
    if isinstance(custom_classes, str):
        custom_classes = [custom_classes]

    if hasattr(model, "set_classes"):
        try:
            model.set_classes(custom_classes)
        except Exception as e:
            print(f"⚠️ Error configurando clases: {e}")

    confidence = float(job_input.get("confidence", 0.25))
    imgsz = int(job_input.get("imgsz", 640))

    # 🔥 INFERENCIA ULTRA RÁPIDA
    with torch.inference_mode():
        results = model.predict(
            source=temp_path,
            imgsz=imgsz,
            conf=confidence,
            device=DEVICE,
            half=True,
            verbose=False
        )

    torch.cuda.synchronize()

    predictions = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = r.names[class_id] if class_id in r.names else f"class_{class_id}"
                predictions.append({
                    "class": class_name,
                    "confidence": round(float(box.conf), 4),
                    "bbox": [round(float(x), 2) for x in box.xyxy[0]]
                })

    os.remove(temp_path)

    return {
        "count": len(predictions),
        "predictions": predictions,
        "classes_used": custom_classes,
        "processing_time": round(time.time() - start_time, 3)
    }

# ==========================================================
# 🚀 INICIO SERVERLESS
# ==========================================================

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
