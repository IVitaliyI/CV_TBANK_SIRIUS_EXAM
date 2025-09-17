FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir python-multipart

COPY best_second_model.pt /app/best.pt
COPY API.py .

EXPOSE 8000

# ==== Запуск ====
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]
