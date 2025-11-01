from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

interpreter = Interpreter(model_path="fatigue_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
def root():
    return {"message": "Driver Drowsiness Backend Running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]["index"])[0][0]

    return {
        "probability": float(pred),
        "state": "closed" if pred < 0.5 else "open"
    }
