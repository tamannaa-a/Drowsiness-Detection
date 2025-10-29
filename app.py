import streamlit as st
import cv2
import numpy as np
import tempfile
import requests
import base64
import os
from io import BytesIO
from pydub.generators import Sine

# Backend API endpoint
BACKEND_URL = "https://drowsiness-detection-1-djgk.onrender.com/predict"

# Function to create a beep sound as base64 (so Streamlit can play it)
def generate_beep_base64():
    sine_wave = Sine(800).to_audio_segment(duration=700)  # 800 Hz, 0.7 sec
    buffer = BytesIO()
    sine_wave.export(buffer, format="mp3")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """

# Streamlit UI
st.set_page_config(page_title="🚗 Drowsiness Detection System", layout="centered")
st.title("🚗 Emotion-Aware Drowsiness Detection System")
st.markdown("### Detect eye state (Open/Closed) to help prevent road accidents.")

# Sidebar for input options
st.sidebar.header("Choose Input Method")
use_camera = st.sidebar.checkbox("Use Camera")

if use_camera:
    img_file_buffer = st.camera_input("Capture Image")
else:
    img_file_buffer = st.file_uploader("Upload an image of your face", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, image)
        tmp_path = tmp.name

    # Send to backend
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
            st.markdown(generate_beep_base64(), unsafe_allow_html=True)
        else:
            st.success("✅ Eyes open. Driver is alert.")
    else:
        st.error("Error connecting to the backend API.")

    os.remove(tmp_path)
