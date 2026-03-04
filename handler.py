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

# Cargar el modelo al inicio
model = None
try:
    print("📥 Cargando modelo YOLO-World...")
    model = YOLOWorld("yolov8x-worldv2.pt")
    if torch.cuda.is_available():
        model.model.to('cuda')
        print("✅ Modelo movido a GPU")
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    raise e

def download_image(image_source):
    """Descarga imagen desde URL o decodifica base64"""
    try:
        if image_source.startswith(('http://', 'https://')):
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(image_source, timeout=30, headers=headers)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        else:
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
    
    print(f"\n🔨 Procesando job: {job_id}")
    
    try:
        job_input = job["input"]
        
        # Validar entrada
        if not job_input.get("image"):
            return {"error": "No se proporcionó ninguna imagen"}
        
        # Obtener la imagen
        image = download_image(job_input["image"])
        temp_path = f"/tmp/temp_image_{job_id}.jpg"
        image.save(temp_path, 'JPEG', quality=95)
        
        # Obtener clases
        custom_classes = job_input.get("classes", ["person"])
        if isinstance(custom_classes, str):
            custom_classes = [custom_classes]
        
        # CORRECCIÓN: Usar la variable global model correctamente
        global model
        try:
            model.set_classes(custom_classes)
        except:
            # Si falla, recargar modelo
            model = YOLOWorld("yolov8x-worldv2.pt")
            model.set_classes(custom_classes)
        
        # Configurar dispositivo
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model.model.to('cuda')
        
        # Parámetros
        confidence = float(job_input.get("confidence", 0.25))
        imgsz = int(job_input.get("imgsz", 640))
        
        # Inferencia
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
                boxes = r.boxes.cpu()
                for box in boxes:
                    class_idx = int(box.cls.item())
                    if class_idx < len(custom_classes):
                        predictions.append({
                            "class": custom_classes[class_idx],
                            "confidence": round(float(box.conf.item()), 4),
                            "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
                        })
        
        # Limpiar
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "predictions": predictions,
            "classes_used": custom_classes,
            "count": len(predictions),
            "processing_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

# Iniciar servidor
print("✅ Handler configurado, iniciando servidor...")
runpod.serverless.start({"handler": handler})
