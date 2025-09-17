import os
import shutil
from ultralytics import YOLO

MODEL_PATH = r"runs/detect/train3/weights/best.pt"
RAW_FOLDER = r"data/raw/data_sirius"
FINAL_DATASET = r"data/FINAL_DATASET"

CONF_THRESHOLD = 0.5

for split in ["train", "val"]:
    os.makedirs(os.path.join(FINAL_DATASET, "images", split), exist_ok=True)
    os.makedirs(os.path.join(FINAL_DATASET, "labels", split), exist_ok=True)

model = YOLO(MODEL_PATH)

print("Разметка новых изображений с помощью модели...")

results = model.predict(
    source=RAW_FOLDER,
    save=False,
    conf=CONF_THRESHOLD,
    stream=True
)

train_img_dir = os.path.join(FINAL_DATASET, "images", "train")
train_lbl_dir = os.path.join(FINAL_DATASET, "labels", "train")

for r in results:
    img_name = os.path.basename(r.path)
    lbl_name = os.path.splitext(img_name)[0] + ".txt"

    shutil.copy(r.path, os.path.join(train_img_dir, img_name))

    dst_lbl = os.path.join(train_lbl_dir, lbl_name)
    with open(dst_lbl, "w") as f:
        for box in r.boxes:
            cls = int(box.cls)
            x_center, y_center, w, h = box.xywhn[0].tolist()
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print("Авторазметка завершена!")
