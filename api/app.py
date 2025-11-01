# api/app.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
import mediapipe as mp
from utils import eye_aspect_ratio

app = FastAPI(title="Drowsiness Inference API (MediaPipe EAR)")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5)

RIGHT_EYE_IDX = [33, 160, 158, 133, 144, 153]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

@app.post("/infer_ear")
async def infer_ear(image: UploadFile = File(...)):
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "unable to decode image"}

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)
    if not res.multi_face_landmarks:
        return {"face": False}

    lm = res.multi_face_landmarks[0]
    h, w, _ = frame.shape
    def coord(i):
        p = lm.landmark[i]
        return np.array([p.x * w, p.y * h])

    r_pts = [coord(i) for i in RIGHT_EYE_IDX]
    l_pts = [coord(i) for i in LEFT_EYE_IDX]
    ear = (eye_aspect_ratio(r_pts) + eye_aspect_ratio(l_pts)) / 2.0
    return {"face": True, "ear": float(ear)}
