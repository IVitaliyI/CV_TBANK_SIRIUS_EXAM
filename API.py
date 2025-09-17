import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from PIL import Image
from ultralytics import YOLO

class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)

class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")

class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")

class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")


app = FastAPI(title="T-Bank Logo Detection API")

# Загружаем обученную модель YOLO (замени на свой путь)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

SUPPORTED_FORMATS = {"jpeg", "jpg", "png", "bmp", "webp"}

@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении
    """

    # Проверка формата
    ext = file.filename.split(".")[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: {SUPPORTED_FORMATS}"
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = model.predict(image, imgsz=640, conf=0.25, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x_min, y_min, x_max, y_max = map(int, box[:4])
                detections.append(
                    Detection(
                        bbox=BoundingBox(
                            x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max
                        )
                    )
                )

        return DetectionResponse(detections=detections)

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": "Processing error", "detail": str(e)},
        )
