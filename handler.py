import runpod
import torch
import os
import requests
import base64
import traceback
import time
import sys
from io import BytesIO
from PIL import Image

print("="*60)
print("🚀 INICIANDO HANDLER DE YOLO-WORLD v2")
print("="*60)

# VERIFICACIÓN DE NUMPY
print("\n🔍 VERIFICANDO DEPENDENCIAS:")
try:
    import numpy as np
    print(f"  ✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"  ❌ NumPy NO disponible: {e}")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'numpy>=1.24.0'])
    import numpy as np
    print(f"  ✅ NumPy instalado: {np.__version__}")

# Verificar CUDA
print("\n🔍 VERIFICANDO CUDA:")
print(f"  CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# IMPORTAR CORRECTAMENTE - USANDO YOLO EN LUGAR DE YOLOWorld
print("\n🔍 CARGANDO YOLO-WORLD:")
try:
    # CORRECCIÓN: Importar YOLO en lugar de YOLOWorld
    from ultralytics import YOLO
    print("  ✅ Ultralytics YOLO importado correctamente")
except Exception as e:
    print(f"  ❌ Error importando ultralytics: {e}")
    traceback.print_exc()
    raise e

# Cargar el modelo - usar YOLO con modelo world
model = None
try:
    print("  📥 Descargando modelo yolov8x-worldv2.pt...")
    # CORRECCIÓN: Usar YOLO con modelo world
    model = YOLO("yolov8x-worldv2.pt")
    
    # Verificar que es un modelo world
    if hasattr(model, 'set_classes'):
        print("  ✅ Modelo YOLO-World detectado correctamente")
    else:
        print("  ⚠️ El modelo cargado no soporta set_classes")
    
    if torch.cuda.is_available():
        model.to('cuda')
        print("  ✅ Modelo movido a GPU")
    print("  ✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"  ❌ Error cargando modelo: {e}")
    traceback.print_exc()
    raise e

print("\n" + "="*60)
print("✅ HANDLER LISTO PARA RECIBIR PETICIONES")
print("="*60 + "\n")

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
    print("-" * 40)
    
    try:
        job_input = job["input"]
        
        # Validar entrada
        if not job_input.get("image"):
            return {"error": "No se proporcionó ninguna imagen"}
        
        # Obtener la imagen
        print("📥 Descargando imagen...")
        image = download_image(job_input["image"])
        temp_path = f"/tmp/temp_image_{job_id}.jpg"
        image.save(temp_path, 'JPEG', quality=95)
        print(f"  - Dimensiones: {image.size}")
        
        # Obtener clases
        custom_classes = job_input.get("classes", ["person"])
        if isinstance(custom_classes, str):
            custom_classes = [custom_classes]
        print(f"🏷️ Clases a detectar ({len(custom_classes)}): {custom_classes}")
        
        # CORRECCIÓN: Configurar modelo con set_classes
        global model
        try:
            if hasattr(model, 'set_classes'):
                model.set_classes(custom_classes)
                print("  ✅ Clases configuradas")
            else:
                print("  ⚠️ El modelo no soporta set_classes, continuando con clases por defecto")
        except Exception as e:
            print(f"  ⚠️ Error configurando clases: {e}")
        
        # Parámetros
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
                boxes = r.boxes.cpu()
                for box in boxes:
                    class_id = int(box.cls.item())
                    class_name = r.names[class_id] if class_id in r.names else f"class_{class_id}"
                    predictions.append({
                        "class": class_name,
                        "confidence": round(float(box.conf.item()), 4),
                        "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
                    })
        
        # Limpiar
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        elapsed_time = round(time.time() - start_time, 2)
        print(f"✅ Completado: {len(predictions)} detecciones en {elapsed_time}s")
        
        return {
            "predictions": predictions,
            "classes_used": custom_classes,
            "count": len(predictions),
            "processing_time": elapsed_time
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

# Iniciar servidor
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
