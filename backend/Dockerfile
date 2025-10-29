FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

# adjust MODEL_PATH env var as needed, or mount model into /app/saved_model
ENV MODEL_PATH=/app/saved_model/drowsiness_model.h5

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
