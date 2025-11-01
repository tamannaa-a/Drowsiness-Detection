import streamlit as st
import cv2
import numpy as np
import requests
import time
import streamlit.components.v1 as components

# ✅ Backend URL (your Render backend)
BACKEND_URL = "https://drowsiness-detection-1-djgk.onrender.com/predict_verbose"

# ✅ Haarcascade for detecting EYES
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# ✅ Browser beep alert
def beep():
    beep_html = """
    <audio autoplay>
        <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
    </audio>
    """
    components.html(beep_html, height=0)

# ✅ Predict eye state using backend
def predict_eye(image):
    # Convert eye crop → bytes
    success, encoded = cv2.imencode(".jpg", image)
    files = {"file": ("eye.jpg", encoded.tobytes(), "image/jpeg")}

    try:
        res = requests.post(BACKEND_URL, files=files).json()
        prob = res["interpretation"]["prob"]
        state = "Open" if prob >= 0.5 else "Closed"
        return state, prob
    except:
        return "Error", 0.0

# ✅ STREAMLIT UI
st.title("🚗 Real-Time Eye State Detection")
st.write("Real-time driver drowsiness detection using webcam")

run = st.toggle("Start Webcam")
frame_window = st.image([])

if run:

    cap = cv2.VideoCapture(0)
    st.info("Webcam activated. Close your eyes to test the alert.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ✅ Detect eyes
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)

        eye_state = "No Eyes Detected"
        prob = 0

        for (x, y, w, h) in eyes[:1]:   # take one eye only (more stable)
            eye = frame[y:y+h, x:x+w]

            # ✅ Resize to model input (224×224)
            eye = cv2.resize(eye, (224, 224))

            # ✅ Predict using backend
            eye_state, prob = predict_eye(eye)

            # Draw bounding box
            color = (0,255,0) if eye_state=="Open" else (0,0,255)
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,f"{eye_state} ({prob:.2f})",(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
            break

        # ✅ Alert if closed
        if eye_state == "Closed":
            st.error("🚨 EYES CLOSED! Driver Drowsy!")
            beep()

        # ✅ Display frame
        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        time.sleep(0.2)

else:
    st.warning("Webcam stopped.")
