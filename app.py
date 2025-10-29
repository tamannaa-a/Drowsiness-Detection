import streamlit as st
import cv2
import numpy as np
import requests
import tempfile
import base64
import time
from io import BytesIO
from PIL import Image
import pygame

# ----------------------------
# CONFIGURATION
# ----------------------------
BACKEND_URL = "https://drowsiness-detection-1-djgk.onrender.com/predict"  # Your backend URL

st.set_page_config(page_title="Driver Drowsiness Detection", layout="centered")

# ----------------------------
# PAGE HEADER
# ----------------------------
st.title("🚗 Real-Time Drowsiness Detection System")
st.markdown("""
This system detects **eye state (Open/Closed)** in real-time using your webcam.  
If eyes are **closed**, it generates an alert sound to help prevent accidents.
""")

# ----------------------------
# SOUND SETUP (alert beep)
# ----------------------------
pygame.mixer.init()
beep_sound = pygame.mixer.Sound("https://actions.google.com/sounds/v1/alarms/beep_short.ogg")

# ----------------------------
# WEBCAM SECTION
# ----------------------------
run = st.checkbox('Start Webcam', value=False)
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

if run:
    st.info("Starting detection... Press 'Stop Webcam' to end.")
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam. Try again.")
            break

        # Convert frame for backend
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        files = {"file": ("frame.png", buf, "image/png")}

        # Send to backend
        try:
            response = requests.post(BACKEND_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                prob = result.get("interpretation", {}).get("prob", 0.5)
                label = "Open" if prob >= 0.5 else "Closed"
                color = (0, 255, 0) if label == "Open" else (0, 0, 255)

                # Draw label on frame
                cv2.putText(frame, f"{label} ({prob*100:.1f}%)", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                # Alert if closed eyes detected
                if label == "Closed":
                    pygame.mixer.Sound.play(beep_sound)
                    st.warning("⚠️ Driver appears drowsy! Wake up!")

            else:
                cv2.putText(frame, "Backend Error", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        except Exception as e:
            cv2.putText(frame, "API Connection Error", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.5)

else:
    st.info("Webcam stopped.")
    cap.release()
