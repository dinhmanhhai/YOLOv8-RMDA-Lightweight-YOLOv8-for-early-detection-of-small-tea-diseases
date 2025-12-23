from ultralytics import YOLO

# Load model custom
model = YOLO("ultralytics/cfg/models/v8/yolov8-mobilenetv4.yaml") 

# Train
model.train(data="data.yaml", epochs=150, imgsz=640, device=0, batch=16, project="runs/train", name="exp")