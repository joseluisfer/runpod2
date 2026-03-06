import runpod
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

print("Loading model...")

model = YOLO("yolov8x-worldv2.pt").to("cuda")

print("Model ready")

def load_image(url):
    r = requests.get(url)
    return Image.open(BytesIO(r.content)).convert("RGB")

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
