# backend/main.py
import os
import io
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Use tensorflow's tflite interpreter
from tensorflow.lite.python.interpreter import Interpreter

# ------------ CONFIG ------------
MODEL_PATH = os.getenv("MODEL_PATH", "saved_model/fatigue_model.tflite")
IMG_SIZE = (224, 224)   # change if your model expects another size
CLASS_NAMES = ["Closed", "Open"]  # Make sure this ordering matches model training
# --------------------------------

app = FastAPI(title="Drowsiness Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your frontend origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    label: str
    score: float

# Load tflite model at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place your tflite model there or set MODEL_PATH env var.")

print("Loading TFLite model from:", MODEL_PATH)
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded. Input details:", input_details, "Output details:", output_details)

def preprocess_image_bytes(image_bytes: bytes):
    """Convert image bytes to a numpy array and preprocess to model input shape."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    # convert BGR->RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    # normalize float32 0..1
    img = img.astype(np.float32) / 255.0
    # Expand dims to (1,H,W,3)
    return np.expand_dims(img, axis=0)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accepts multipart file upload (image). Returns predicted label and score.
    """
    contents = await file.read()
    img = preprocess_image_bytes(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # set tensor
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    # handle different output shapes (softmax, logits, etc.)
    probs = np.squeeze(output_data)
    # if single output (logit) -> convert with sigmoid
    if probs.ndim == 0 or probs.size == 1:
        # binary single-output -> apply sigmoid
        prob = float(1.0 / (1.0 + np.exp(-probs)))
        # map to classes: treat prob as "Open" probability by convention (adjust if needed)
        probs_arr = np.array([1 - prob, prob])
    else:
        # assume probs is array-like e.g. [p_closed, p_open] or [p_open, p_closed]
        probs_arr = probs
        # if outputs are logits, apply softmax
        if np.any(probs_arr < 0) or probs_arr.sum() != 1.0:
            exp = np.exp(probs_arr - np.max(probs_arr))
            probs_arr = exp / exp.sum()

    idx = int(np.argmax(probs_arr))
    score = float(probs_arr[idx])

    label = CLASS_NAMES[idx]
    return {"label": label, "score": score}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
