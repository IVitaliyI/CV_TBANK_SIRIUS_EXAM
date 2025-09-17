import os
import cv2
import requests

API_URL = r"http://localhost:8000/detect"
IMAGES_DIR = "data/raw/data_sirius"  # папка с тестовыми изображениями

# Список изображений
image_files = [f for f in os.listdir(IMAGES_DIR)
               if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]

idx = 0
while idx < len(image_files):
    image_path = os.path.join(IMAGES_DIR, image_files[idx])
    
    # Отправляем изображение на сервер
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(API_URL, files=files)
    
    if response.status_code != 200:
        print(f"Ошибка {response.status_code}: {response.text}")
        idx += 1
        continue
    
    data = response.json()
    
    # Загружаем изображение
    img = cv2.imread(image_path)
    
    # Рисуем bbox
    for det in data.get("detections", []):
        bbox = det["bbox"]
        cv2.rectangle(img,
                      (bbox["x_min"], bbox["y_min"]),
                      (bbox["x_max"], bbox["y_max"]),
                      (0, 0, 255), 2)
    
    # Показываем изображение
    cv2.imshow("Detection", img)
    
    # Ждём клавишу
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("n"):
        idx += 1

cv2.destroyAllWindows()
