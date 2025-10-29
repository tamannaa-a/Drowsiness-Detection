# backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc libglib2.0-0 libsm6 libxrender1 libxext6 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=/app/saved_model/fatigue_model.tflite
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
