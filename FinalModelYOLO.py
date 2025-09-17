from ultralytics import YOLO

model: YOLO = YOLO("yolo11s.pt")

results = model.train(
    data=r"finalData.yaml",    
    epochs=5,
    imgsz=640,
    batch=6,
    workers=8,
    device=0
)

print("Лучшие веса сохранены в:", results.save_dir)