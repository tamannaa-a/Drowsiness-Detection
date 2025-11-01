import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

st.set_page_config(layout="wide")

# ------------------------
# Load TFLite model
# ------------------------
interpreter = Interpreter(model_path="fatigue_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------
# Load MediaPipe
# ------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ------------------------
# Alarm sound
# ------------------------
def beep_alarm():
    wave_obj = sa.WaveObject.from_wave_file("beep.wav")  # You can put beep.wav in the folder
    play_obj = wave_obj.play()

# ------------------------
# Helper: Crop eye region from landmarks
# ------------------------
def crop_eye_region(frame, landmarks, eye_indices):
    h, w, _ = frame.shape
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]

    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]

    min_x, max_x = max(min(x_coords) - 5, 0), min(max(x_coords) + 5, w)
    min_y, max_y = max(min(y_coords) - 5, 0), min(max(y_coords) + 5, h)

    eye_img = frame[min_y:max_y, min_x:max_x]

    if eye_img.size == 0:
        return None

    eye_img = cv2.resize(eye_img, (224, 224))
    eye_img = eye_img.astype("float32") / 255.0
    return eye_img

# ------------------------
# Eye landmark indices for MediaPipe FaceMesh
# ------------------------
LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466]

# ------------------------
# Prediction function
# ------------------------
def predict_eye_state(eye_img):
    eye_img = np.expand_dims(eye_img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], eye_img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output[0]  # sigmoid probability

# ------------------------
# Streamlit UI
# ------------------------
st.title("🚗 Real-Time Driver Drowsiness Detection")
st.markdown("**Eye Open/Closed Detection using TFLite + MediaPipe**")

run_btn = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = None

# ------------------------
# Main loop
# ------------------------
if run_btn:
    cap = cv2.VideoCapture(0)

    closed_frames = 0
    CLOSED_THRESHOLD = 10  # alarm after 10 consecutive closed predictions

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        eye_state_text = "Detecting..."
        eye_color = (0, 255, 0)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark

            left_eye = crop_eye_region(frame, landmarks, LEFT_EYE)
            right_eye = crop_eye_region(frame, landmarks, RIGHT_EYE)

            if left_eye is not None:
                p_left = predict_eye_state(left_eye)
            else:
                p_left = 1  

            if right_eye is not None:
                p_right = predict_eye_state(right_eye)
            else:
                p_right = 1  

            avg_prob = (p_left + p_right) / 2

            if avg_prob < 0.5:
                eye_state_text = "EYES CLOSED"
                eye_color = (0, 0, 255)
                closed_frames += 1
            else:
                eye_state_text = "EYES OPEN"
                eye_color = (0, 255, 0)
                closed_frames = 0

            # Trigger alarm
            if closed_frames > CLOSED_THRESHOLD:
                beep_alarm()

            cv2.putText(frame, f"{eye_state_text} ({avg_prob:.2f})",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, eye_color, 3)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    if cap:
        cap.release()
