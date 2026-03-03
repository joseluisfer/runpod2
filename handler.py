import runpod
from ultralytics import YOLOWorld
import torch
import os
import requests
import base64
from io import BytesIO
from PIL import Image
import traceback

print("🚀 Iniciando handler de YOLO-World...")

# Determinar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Dispositivo disponible: {device}")

# Cargar modelo y forzar dispositivo
try:
    print("📥 Cargando modelo YOLO-World v2-X...")
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.to(device)  # Forzar modelo completo a GPU si está disponible
    print(f"✅ Modelo cargado en {device}")
    
    # Verificar que todos los parámetros están en el mismo dispositivo
    if device == "cuda":
        param_device = next(model.parameters()).device
        print(f"🔍 Parámetros del modelo en: {param_device}")
except Exception as e:
    print(f"❌ Error fatal cargando modelo: {e}")
    traceback.print_exc()
    raise e

def download_image(image_source):
    """Descarga imagen desde URL o decodifica base64"""
    try:
        if image_source.startswith(('http://', 'https://')):
            print(f"📥 Descargando imagen desde URL...")
            response = requests.get(image_source, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        else:
            print(f"📥 Decodificando imagen desde base64...")
            # Limpiar posible prefijo data:image
            if ',' in image_source:
                image_source = image_source.split(',')[1]
            image_data = base64.b64decode(image_source)
            return Image.open(BytesIO(image_data)).convert('RGB')
    except Exception as e:
        raise Exception(f"Error al procesar la imagen: {e}")

def handler(job):
    job_id = job.get('id', 'unknown')
    print(f"\n🔨 Procesando job {job_id}")
    
    try:
        job_input = job["input"]
        
        # 1. Validar entrada
        if not job_input.get("image"):
            return {"error": "No se proporcionó ninguna imagen"}
        
        # 2. Obtener y preparar imagen
        image_source = job_input["image"]
        image = download_image(image_source)
        
        # Guardar temporalmente (YOLO prefiere archivos locales)
        temp_path = f"/tmp/temp_image_{job_id}.jpg"
        image.save(temp_path)
        print(f"💾 Imagen guardada en {temp_path}")
        
        # 3. Configurar clases dinámicas
        custom_classes = job_input.get("classes", ["person", "car", "dog"])
        if isinstance(custom_classes, str):
            custom_classes = [custom_classes]
        print(f"🏷️ Clases solicitadas: {custom_classes}")
        
        # Aplicar clases y luego forzar modelo al dispositivo correcto
        model.set_classes(custom_classes)
        model.to(device)  # <--- CLAVE: después de set_classes, mover todo a GPU
        
        # Verificar dispositivo después de set_classes
        if device == "cuda":
            param_device = next(model.parameters()).device
            print(f"🔍 Parámetros después de set_classes: {param_device}")
        
        # 4. Parámetros de inferencia
        confidence = float(job_input.get("confidence", 0.25))
        imgsz = int(job_input.get("imgsz", 640))
        
        # 5. Inferencia
        print(f"🔍 Ejecutando inferencia (conf={confidence}, imgsz={imgsz})...")
        results = model.predict(
            source=temp_path,
            imgsz=imgsz,
            conf=confidence,
            device=device,  # Explícito
            verbose=False
        )
        
        # 6. Procesar resultados (mover a CPU para serialización)
        predictions = []
        for r in results:
            if r.boxes is not None:
                # Mover boxes a CPU
                boxes = r.boxes.cpu()
                for box in boxes:
                    class_idx = int(box.cls.item())
                    if class_idx < len(custom_classes):
                        predictions.append({
                            "class": custom_classes[class_idx],
                            "confidence": round(float(box.conf.item()), 4),
                            "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
                        })
        
        print(f"✅ {len(predictions)} detecciones encontradas")
        
        # Limpiar archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "predictions": predictions,
            "classes_used": custom_classes,
            "count": len(predictions),
            "device_used": device
        }
        
    except Exception as e:
        print(f"❌ Error en job {job_id}: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "trace": traceback.format_exc()}

# Iniciar servidor
print("✅ Handler listo, iniciando servidor RunPod...")
runpod.serverless.start({"handler": handler})
