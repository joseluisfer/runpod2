import runpod
from ultralytics import YOLOWorld
import torch
import os
import requests
import base64
from io import BytesIO
from PIL import Image

# Cargar el modelo en la memoria de la GPU al arrancar el contenedor
try:
    print("Iniciando carga del modelo YOLO-World v2-X...")
    model = YOLOWorld("yolov8x-worldv2.pt")
    print("Modelo cargado. Listo para recibir clases dinámicamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    raise e

def download_image(image_source):
    """Descarga imagen desde URL o decodifica base64"""
    try:
        if image_source.startswith('http'):
            response = requests.get(image_source, timeout=30)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Asumimos que es base64
            image_data = base64.b64decode(image_source)
            return Image.open(BytesIO(image_data)).convert('RGB')
    except Exception as e:
        raise Exception(f"Error al procesar la imagen: {e}")

def handler(job):
    try:
        job_input = job["input"]
        
        # Validar entrada
        if not job_input.get("image"):
            return {"error": "No se proporcionó ninguna imagen"}
        
        # Obtener la imagen
        image_source = job_input["image"]
        image = download_image(image_source)
        
        # Guardar temporalmente para YOLO
        temp_path = "/tmp/temp_image.jpg"
        image.save(temp_path)
        
        # Obtener clases
        custom_classes = job_input.get("classes", ["object"])
        if not isinstance(custom_classes, list):
            custom_classes = [custom_classes]
        
        # Configurar modelo
        model.set_classes(custom_classes)
        
        # Parámetros de inferencia
        confidence = float(job_input.get("confidence", 0.25))
        imgsz = int(job_input.get("imgsz", 1280))
        
        # Inferencia
        results = model.predict(
            source=temp_path,
            imgsz=imgsz,
            conf=confidence,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False
        )
        
        # Formatear salida
        predictions = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    class_idx = int(box.cls)
                    if class_idx < len(custom_classes):
                        predictions.append({
                            "class": custom_classes[class_idx],
                            "confidence": round(float(box.conf), 4),
                            "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
                        })
        
        # Limpiar archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "predictions": predictions,
            "classes_used": custom_classes,
            "count": len(predictions)
        }
        
    except Exception as e:
        return {"error": str(e)}

# Iniciar el servidor
runpod.serverless.start({"handler": handler})