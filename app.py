import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import requests
import simpleaudio as sa

BACKEND_URL = "http://localhost:8000/predict"

st.set_page_config(layout="wide")
st.title("🚗 Real-Time Driver Drowsiness Detection (Frontend)")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466]

def beep_alarm():
    wave_obj = sa.WaveObject.from_wave_file("beep.wav")
    wave_obj.play()

def crop_eye(frame, landmarks, indexes):
    h, w, _ = frame.shape
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indexes]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x1, x2 = max(min(xs)-5,0), min(max(xs)+5,w)
    y1, y2 = max(min(ys)-5,0), min(max(ys)+5,h)

    eye = frame[y1:y2, x1:x2]
    if eye.size == 0:
        return None

    return eye

if run:
    cap = cv2.VideoCapture(0)

    closed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        eye_state = "Detecting..."
        eye_color = (0, 255, 0)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            left_eye = crop_eye(frame, lm, LEFT_EYE)
            right_eye = crop_eye(frame, lm, RIGHT_EYE)

            eyes = []
            if left_eye is not None: eyes.append(left_eye)
            if right_eye is not None: eyes.append(right_eye)

            states = []

            for eye_img in eyes:
                _, buf = cv2.imencode(".jpg", eye_img)
                files = {"file": ("eye.jpg", buf.tobytes(), "image/jpeg")}
                r = requests.post(BACKEND_URL, files=files)
                prediction = r.json()

                states.append(prediction["state"])

            if "closed" in states:
                closed_count += 1
                eye_state = "EYES CLOSED"
                eye_color = (0, 0, 255)

                if closed_count > 10:
                    beep_alarm()

            else:
                closed_count = 0
                eye_state = "EYES OPEN"
                eye_color = (0, 255, 0)

        cv2.putText(
            frame, eye_state, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, eye_color, 3
        )

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    st.write("✅ Camera stopped")
