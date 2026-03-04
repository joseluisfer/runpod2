def handler(job):
    """Maneja cada petición al endpoint con corrección de dispositivo"""
    job_id = job.get('id', 'unknown')
    print(f"\n🔨 Procesando job {job_id}")
    
    try:
        job_input = job["input"]
        
        # Validar entrada
        if not job_input.get("image"):
            return {"error": "No se proporcionó ninguna imagen"}
        
        # Forzar el modelo a GPU si está disponible
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {device}")
        
        # Asegurar que el modelo está en el dispositivo correcto
        model.model.to(device)
        
        # Obtener la imagen
        image_source = job_input["image"]
        
        # IMPORTANTE: Pasar la imagen como archivo, no como URL
        import requests
        from PIL import Image
        from io import BytesIO
        
        # Descargar imagen primero
        response = requests.get(image_source, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Guardar temporalmente
        temp_path = f"/tmp/temp_image_{job_id}.jpg"
        image.save(temp_path)
        print(f"💾 Imagen guardada en {temp_path}")
        
        # Obtener clases
        custom_classes = job_input.get("classes", ["person", "car"])
        if isinstance(custom_classes, str):
            custom_classes = [custom_classes]
        print(f"🏷️ Clases: {custom_classes}")
        
        # Configurar modelo (asegurar que está en el device correcto)
        model.set_classes(custom_classes)
        
        # Parámetros de inferencia
        confidence = float(job_input.get("confidence", 0.25))
        imgsz = int(job_input.get("imgsz", 640))
        
        # Inferencia con manejo explícito de dispositivo
        results = model.predict(
            source=temp_path,  # Usar ruta de archivo, no URL
            imgsz=imgsz,
            conf=confidence,
            device=device,
            verbose=False
        )
        
        # Procesar resultados...
        predictions = []
        for r in results:
            if r.boxes is not None:
                # Mover boxes a CPU para procesamiento seguro
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
            "device": device
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
