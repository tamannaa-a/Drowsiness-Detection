from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import cv2
import os
import io

app = FastAPI()

# ✅ Correct path to your model
MODEL_PATH = "saved_model/fatigue_model.tflite"

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # ✅ Preprocess: match training size (adjust if needed)
        IMG_SIZE = (224, 224)  # or 128x128 depending on your training
        img_resized = cv2.resize(img, IMG_SIZE)
        img_array = img_resized.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0][0]

        # ✅ Interpret result
        # Model likely returns sigmoid: 0=open, 1=closed
        eye_state = "Closed" if output_data > 0.5 else "Open"
        confidence = round(float(output_data * 100), 2) if eye_state == "Closed" else round(float((1 - output_data) * 100), 2)

        return JSONResponse({
            "eye_state": eye_state,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "eye_state": "unknown",
            "confidence": 0
        })
