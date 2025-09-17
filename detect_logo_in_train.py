import os
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import shutil
import numpy as np

# # Загрузка модели
model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "GroundingDINO/weights/groundingdino_swint_ogc.pth"
)
prompts = r"shield bank logo ."
prompts = r"T shaped logo ."
prompts = r"""logo . Shield . T ."""

BOX_THRESHOLD = 0.2
TEXT_THRESHOLD = 0.3

raw_folder = "data/raw/data_sirius/"
output_folder = r"data/preprocessed/"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
    os.mkdir(output_folder)
else:
    os.mkdir(output_folder)

def interactive_check_matplotlib_gui(image: np.ndarray, boxes, logits, phrases) -> list:
    cv2.namedWindow('fot', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('fot', 1920, 1080)
    
    annotated_image = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
    
    cv2.imshow('fot', annotated_image)
    key = cv2.waitKey(0)
    
    cv2.destroyAllWindows()
        

def model_predict(image: torch.Tensor) -> torch.Tensor:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    return predict(
        model=model,
        image=image,
        caption=prompts,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
def work():
    for filename in os.listdir(raw_folder):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(raw_folder, filename)
        image_source, image = load_image(image_path)

        filename_no_ext = os.path.splitext(filename)[0]
        save_path_txt = os.path.join(output_folder, filename_no_ext + ".txt")
        save_path_img = os.path.join(output_folder, filename_no_ext + ".jpg")

        interactive_check_yolo(
            image_source, *model_predict(image),
            save_path_txt=save_path_txt,
            save_path_img=save_path_img
        )


work()