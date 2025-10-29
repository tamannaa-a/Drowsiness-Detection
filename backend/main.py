# backend/main.py
import os
import io
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# Config
MODEL_PATH = os.getenv("MODEL_PATH", "saved_model/drowsiness_model.h5")
IMG_SIZE = (224, 224)     # must match model training size
CLASS_NAMES = ["Closed", "Open"]  # ordering must match training

app = FastAPI(title="Drowsiness Detection API")

# Allow CORS from any origin (for demo). For production set specific origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    label: str
    score: float

# Load model at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place your model there or set MODEL_PATH env var.")

print("Loading model from", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Model loaded.")

def preprocess_image_bytes(image_bytes: bytes):
    """Read image bytes into a numpy array and preprocess for model."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts multipart file (JPEG/PNG). Returns label and score.
    """
    contents = await file.read()
    img = preprocess_image_bytes(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    preds = model.predict(img)[0]  # e.g. [0.12, 0.88]
    idx = int(np.argmax(preds))
    return {"label": CLASS_NAMES[idx], "score": float(preds[idx])}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
