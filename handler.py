import runpod
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

print("Loading model...")

model = YOLO("yolov8x-worldv2.pt").to("cuda")

print("Model ready")

def load_image(url):

        # caso 1: URL
    if image_input.startswith("http"):
        response = requests.get(image_input)
        return Image.open(BytesIO(response.content)).convert("RGB")

    # caso 2: base64
    else:
        img_bytes = base64.b64decode(image_input)
        return Image.open(BytesIO(img_bytes)).convert("RGB")

def handler(job):

    url = job["input"]["image"]
    classes = job["input"].get("classes", ["person"])

    if hasattr(model, "set_classes"):
        model.set_classes(classes)

    img = load_image(url)

    results = model(img, imgsz=640)

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            })

    return {"detections": detections}

runpod.serverless.start({"handler": handler})

