import runpod
from ultralytics import YOLOWorld
import torch
import os
import requests
import base64
from io import BytesIO
from PIL import Image
import traceback
import time

print("="*50)
print("🚀 INICIANDO HANDLER DE YOLO-WORLD")
print("="*50)

# Verificar CUDA
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Cargar el modelo al inicio con reintentos
model = None
max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"📥 Intento {attempt + 1}/{max_retries}: Cargando modelo YOLO-World...")
        model = YOLOWorld("yolov8x-worldv2.pt")
        
        # Mover modelo a GPU si está disponible
        if torch.cuda.is_available():
            model.model.to('cuda')
            print("✅ Modelo movido a GPU")
        
        print("✅ Modelo cargado exitosamente")
        break
    except Exception as e:
        print(f"❌ Error cargando modelo (intento {attempt + 1}): {e}")
        if attempt < max_retries - 1:
            print("⏳ Esperando 5 segundos antes de reintentar...")
            time.sleep(5)
        else:
            print("❌ No se pudo cargar el modelo después de múltiples intentos")
            raise e

def download_image(image_source):
    """Descarga imagen desde URL o decodifica base64"""
    try:
        if image_source.startswith(('http://', 'https://')):
            print(f"📥 Descargando imagen desde URL...")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(image_source, timeout=30, headers=headers)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        else:
            print(f"📥 Decodificando imagen desde base64...")
            # Limpiar el string base64 si tiene prefijo
            if 'base64,' in image_source:
                image_source = image_source.split('base64,')[1]
            image_data = base64.b64decode(image_source)
            return Image.open(BytesIO(image_data)).convert('RGB')
    except Exception as e:
        raise Exception(f"Error al procesar la imagen: {e}")

def handler(job):
    """Maneja cada petición al endpoint"""
    job_id = job.get('id', 'unknown')
    start_time = time.time()
    
    print("\n" + "="*50)
    print(f"🔨 Procesando job: {job_id}")
    print("="*50)
    
    try:
        job_input = job["input"]
        
        # Validar entrada
        if not job_input.get("image"):
            return {"error": "No se proporcionó ninguna imagen"}
        
        # Obtener la imagen
        image_source = job_input["image"]
        image = download_image(image_source)
        
        # Guardar temporalmente
        temp_path = f"/tmp/temp_image_{job_id}.jpg"
        image.save(temp_path, 'JPEG', quality=95)
        print(f"💾 Imagen guardada: {temp_path} ({image.size})")
        
        # Obtener clases
        custom_classes = job_input.get("classes", ["person", "car", "dog", "cat"])
        if isinstance(custom_classes, str):
            custom_classes = [custom_classes]
        print(f"🏷️ Clases a detectar ({len(custom_classes)}): {custom_classes}")
        
        # Configurar modelo
        try:
            model.set_classes(custom_classes)
            print("✅ Clases configuradas")
        except Exception as e:
            print(f"⚠️ Error configurando clases: {e}")
            # Intentar recargar modelo si falla
            global model
            model = YOLOWorld("yolov8x-worldv2.pt")
            model.set_classes(custom_classes)
        
        # Asegurar que el modelo está en el dispositivo correcto
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model.model.to('cuda')
        
        # Parámetros de inferencia
        confidence = float(job_input.get("confidence", 0.25))
        imgsz = int(job_input.get("imgsz", 640))
        print(f"⚙️ Parámetros: conf={confidence}, imgsz={imgsz}, device={device}")
        
        # Inferencia
        print("🔍 Ejecutando inferencia...")
        results = model.predict(
            source=temp_path,
            imgsz=imgsz,
            conf=confidence,
            device=device,
            verbose=False
        )
        
        # Procesar resultados
        predictions = []
        for r in results:
            if r.boxes is not None:
                # Mover a CPU para procesamiento seguro
                boxes = r.boxes.cpu()
                for box in boxes:
                    class_idx = int(box.cls.item())
                    if class_idx < len(custom_classes):
                        predictions.append({
                            "class": custom_classes[class_idx],
                            "confidence": round(float(box.conf.item()), 4),
                            "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
                        })
        
        # Limpiar archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        elapsed_time = round(time.time() - start_time, 2)
        print(f"✅ Job completado en {elapsed_time}s - {len(predictions)} detecciones")
        
        return {
            "predictions": predictions,
            "classes_used": custom_classes,
            "count": len(predictions),
            "processing_time": elapsed_time,
            "device": device,
            "job_id": job_id
        }
        
    except Exception as e:
        print(f"❌ Error en job {job_id}: {str(e)}")
        traceback.print_exc()
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "job_id": job_id
        }

# Iniciar el servidor
print("\n" + "="*50)
print("✅ Handler configurado correctamente")
print("🚀 Iniciando servidor RunPod...")
print("="*50 + "\n")

runpod.serverless.start({"handler": handler})