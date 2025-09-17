from ultralytics import YOLO

model: YOLO = YOLO("yolo11n.pt")

results = model.train(
    data="data.yaml",    
    epochs=50,
    imgsz=640,
    batch=4,
    workers=4,
    device=0
)

print("Лучшие веса сохранены в:", results.save_dir)