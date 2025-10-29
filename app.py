import streamlit as st
import cv2
import numpy as np
import tempfile
import requests
import base64
import os
from pydub import AudioSegment
from pydub.playback import play
import io

# Backend API endpoint
BACKEND_URL = "https://drowsiness-detection-1-djgk.onrender.com/predict"

# Function to play beep sound
def play_beep():
    # Create a beep sound dynamically using pydub
    tone = AudioSegment.sine(frequency=800, duration=500)  # 800 Hz tone for 0.5 sec
    play(tone)

# Page title
st.set_page_config(page_title="🚗 Drowsiness Detection System", layout="centered")
st.title("🚗 Emotion-Aware Drowsiness Detection System")
st.markdown("### Detect eye state (Open/Closed) to help prevent road accidents.")

# File uploader for image or webcam
st.sidebar.header("Upload Image or Use Camera")
use_camera = st.sidebar.checkbox("Use Camera")

if use_camera:
    img_file_buffer = st.camera_input("Capture Image")
else:
    img_file_buffer = st.file_uploader("Upload an image of your face", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display image
    st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # Save to temp file for sending to backend
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, image)
        tmp_path = tmp.name

    # Send to backend API
    with open(tmp_path, "rb") as f:
        files = {"file": f}
        response = requests.post(BACKEND_URL, files=files)

    if response.status_code == 200:
        result = response.json()
        state = result.get("eye_state", "unknown").capitalize()
        confidence = result.get("confidence", "0")

        st.success(f"**Detected Eye State:** {state}")
        st.info(f"**Confidence:** {confidence}%")

        if state.lower() == "closed":
            st.warning("⚠️ Driver appears drowsy! Beep alert activated.")
            play_beep()
        else:
            st.success("✅ Eyes open. Driver is alert.")
    else:
        st.error("Error connecting to the backend API. Please try again later.")

    os.remove(tmp_path)
