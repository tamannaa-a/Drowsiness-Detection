import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from datetime import datetime
import csv
import streamlit.components.v1 as components
import requests


# -----------------------------
# Configuration
# -----------------------------
BACKEND_URL = "https://drowsiness-detection-1-djgk.onrender.com/predict"
MODEL_PATH = "fatigue_model.tflite"  # Optional local fallback

# -----------------------------
# Helper functions
# -----------------------------
def play_beep():
    """Play a short browser beep using HTML."""
    beep_html = """
    <audio autoplay>
        <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
    </audio>
    """
    components.html(beep_html, height=0)

def log_alert():
    """Log drowsiness detection events."""
    with open("drowsiness_log.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Drowsiness detected"])

def predict_eye_state(frame):
    """Send frame to backend for prediction."""
    try:
        _, img_encoded = cv2.imencode(".jpg", frame)
        files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
        import requests
        response = requests.post(BACKEND_URL, files=files)
        if response.status_code == 200:
            data = response.json()
            prob = data.get("interpretation", {}).get("prob", 0.5)
            label = "Open" if prob >= 0.5 else "Closed"
            return label, prob
        else:
            return "Unknown", 0
    except Exception as e:
        print("Backend error:", e)
        return "Error", 0

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Emotion Aware Driving Alert System", layout="wide")
st.title("🚗 Emotion-Aware Driving Alert System")
st.markdown("### Real-time Drowsiness Detection using Eye State Analysis")

st.sidebar.header("🔧 Settings")
confidence_threshold = st.sidebar.slider("Drowsiness Alert Threshold", 0.0, 1.0, 0.5, 0.05)

stframe = st.empty()
alert_placeholder = st.empty()

run = st.toggle("Start Camera")

if run:
    cap = cv2.VideoCapture(0)
    st.markdown("🟢 **Camera Active — Scanning for Drowsiness...**")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected.")
            break

        # Preprocess for prediction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (224, 224))

        # Get prediction
        label, prob = predict_eye_state(resized)

        # Display on frame
        color = (0, 255, 0) if label == "Open" else (0, 0, 255)
        cv2.putText(frame, f"Eye State: {label} ({prob:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        stframe.image(frame, channels="RGB")

        # Trigger alerts if drowsy
        if label == "Closed" and prob < confidence_threshold:
            alert_placeholder.error("🚨 EYES CLOSED! DRIVER DROWSY! 🚨")
            play_beep()
            log_alert()
        else:
            alert_placeholder.empty()

        time.sleep(0.1)

    cap.release()
else:
    st.info("👆 Click 'Start Camera' to begin monitoring.")

st.markdown("---")
st.caption("Developed by Tamanna Vaikkath | Emotion Aware Driving Alert System | Powered by Streamlit & TensorFlow Lite")
