# app.py  (debug-friendly TFLite API)
import os
import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drowsiness-backend")

app = FastAPI(title="Drowsiness TFLite Debug API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config via env
MODEL_PATH = os.getenv("MODEL_PATH", "saved_model/fatigue_model.tflite")
IMG_SIZE_ENV = os.getenv("IMG_SIZE", "224,224")    # "width,height"
CLASS_NAMES_ENV = os.getenv("CLASS_NAMES", "Closed,Open")  # comma-separated
NORMALIZE = os.getenv("NORMALIZE", "1").lower() in ("1", "true", "yes")

try:
    IMG_W, IMG_H = [int(x) for x in IMG_SIZE_ENV.split(",")]
except Exception:
    IMG_W, IMG_H = (224, 224)

CLASS_NAMES = [s.strip() for s in CLASS_NAMES_ENV.split(",")]
logger.info(f"CONFIG: MODEL_PATH={MODEL_PATH} IMG_SIZE={(IMG_W, IMG_H)} CLASS_NAMES={CLASS_NAMES} NORMALIZE={NORMALIZE}")

if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Load tflite interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info(f"Loaded TFLite model. input_details={input_details}, output_details={output_details}")

def preprocess_image_bytes(image_bytes: bytes):
    """Decode bytes -> cv2 image -> resize -> normalize -> batch"""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "can't decode image"
    # Convert BGR->RGB if model expects RGB (most do). We'll keep as RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize
    img_resized = cv2.resize(img, (IMG_W, IMG_H))
    # Optional normalization
    if NORMALIZE:
        img_resized = img_resized.astype(np.float32) / 255.0
    else:
        img_resized = img_resized.astype(np.float32)
    # Expand dims
    batched = np.expand_dims(img_resized, axis=0)
    return batched, None

def safe_softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/info")
def info():
    """Return model I/O details to help debugging."""
    try:
        info = {
            "model_path": MODEL_PATH,
            "img_size": [IMG_W, IMG_H],
            "class_names": CLASS_NAMES,
            "normalize": NORMALIZE,
            "input_details": [{
                "index": d["index"],
                "shape": d["shape"].tolist() if hasattr(d["shape"], "tolist") else list(d["shape"]),
                "dtype": str(d["dtype"]),
                "quantization": d.get("quantization")
            } for d in input_details],
            "output_details": [{
                "index": d["index"],
                "shape": d["shape"].tolist() if hasattr(d["shape"], "tolist") else list(d["shape"]),
                "dtype": str(d["dtype"]),
                "quantization": d.get("quantization")
            } for d in output_details]
        }
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_verbose")
async def predict_verbose(file: UploadFile = File(...)):
    """
    Returns raw output tensor and preprocessing diagnostics.
    Use this to inspect what the TFLite model is returning for a given image.
    """
    try:
        content = await file.read()
        inp, err = preprocess_image_bytes(content)
        if err:
            raise HTTPException(status_code=400, detail=err)

        # Log input stats
        logger.info(f"Input tensor shape: {inp.shape}, min={inp.min():.6f}, max={inp.max():.6f}, mean={inp.mean():.6f}")

        # Set interpreter input
        interpreter.set_tensor(input_details[0]["index"], inp.astype(input_details[0]["dtype"]))
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_details[0]["index"])
        # Convert to plain list for JSON
        raw_list = np.array(raw_out).tolist()

        # Also attempt a best-effort interpretation:
        out = np.array(raw_out).squeeze()
        logger.info(f"Raw output array (squeezed): {out} dtype={out.dtype} shape={out.shape}")

        # If single scalar -> sigmoid-like
        if np.isscalar(out) or out.shape == () or (hasattr(out.shape,'__len__') and len(out.shape)==0):
            val = float(out)
            # assume sigmoid -> prob for class 1 (Open or Closed depending on training)
            prob = 1.0 / (1.0 + float(np.exp(-val)))
            interpretation = {"type": "sigmoid_scalar", "prob": prob, "mapping_note": "prob maps to class index 1 by convention; adjust CLASS_NAMES env if reversed"}
        else:
            # vector output
            out_vec = np.array(out).astype(float)
            # if sums approx 1 -> assume softmax probabilities
            s = out_vec.sum()
            if np.isclose(s, 1.0, atol=1e-2):
                probs = out_vec
                interpretation = {"type": "softmax_like", "probs": probs.tolist()}
            else:
                # apply softmax as best effort
                probs = safe_softmax(out_vec)
                interpretation = {"type": "softmax_applied", "probs": probs.tolist()}

        return {
            "raw_output": raw_list,
            "interpretation": interpretation
        }

    except Exception as e:
        logger.exception("Error in predict_verbose")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Safe predict endpoint: returns eye_state and confidence.
    Uses heuristics:
      - If model outputs scalar: treat as sigmoid -> value>0.5 => CLASS_NAMES[1]
      - If model outputs vector: apply softmax and take argmax
    """
    try:
        content = await file.read()
        inp, err = preprocess_image_bytes(content)
        if err:
            raise HTTPException(status_code=400, detail=err)

        # Log input stats
        logger.info(f"Predict input shape: {inp.shape}, min={inp.min():.6f}, max={inp.max():.6f}")

        interpreter.set_tensor(input_details[0]["index"], inp.astype(input_details[0]["dtype"]))
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_details[0]["index"])
        out = np.array(raw_out).squeeze()
        logger.info(f"Raw model output (squeezed): {out}")

        # Interpret
        if np.isscalar(out) or out.shape == () or (hasattr(out.shape,'__len__') and len(out.shape)==0):
            # sigmoid scalar
            val = float(out)
            prob_class1 = 1.0 / (1.0 + float(np.exp(-val)))
            # Decide mapping: by convention CLASS_NAMES[1] is prob_class1
            idx = 1 if prob_class1 > 0.5 else 0
            confidence = float(prob_class1) if idx == 1 else float(1.0 - prob_class1)
            mapped_label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
            logger.info(f"Interpreted scalar: val={val:.6f} prob_class1={prob_class1:.4f} -> idx={idx} label={mapped_label}")
        else:
            vec = np.array(out).astype(float)
            s = vec.sum()
            if np.isclose(s, 1.0, atol=1e-2):
                probs = vec
            else:
                probs = safe_softmax(vec)
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            mapped_label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
            logger.info(f"Interpreted vector: probs={probs} -> idx={idx} label={mapped_label}")

        return JSONResponse({"eye_state": mapped_label, "confidence": round(confidence * 100.0, 2)})

    except Exception as e:
        logger.exception("Error in /predict")
        raise HTTPException(status_code=500, detail=str(e))
