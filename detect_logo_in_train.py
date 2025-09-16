import os
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import shutil

# Загрузка модели
model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "GroundingDINO/weights/groundingdino_swint_ogc.pth"
)

prompts = r"shield bank logo . Letter T . Bank logo ."


BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.2

train_folder = "data/raw/data_sirius/"
output_folder = "data/annotated"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(train_folder):
    if not filename.lower().endswith((".jpg", ".png")):
        continue
    
    image_path = os.path.join(train_folder, filename)
    image_source, image = load_image(image_path)
    
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompts,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    annotated_frame = annotate(
        image_source=image_source,
        boxes=boxes,
        logits=logits,
        phrases=phrases
    )
    cv2.imwrite(os.path.join(output_folder, filename), annotated_frame)
