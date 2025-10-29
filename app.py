import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import time
import base64
from PIL import Image

# ---------------------------
# 🧠 Load the TFLite model
# ---------------------------
MODEL_PATH = "fatigue_model.tflite"

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------------------
# 🚀 Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title="Emotion-Aware Driving Alert System", layout="centered")
st.title("🚗 Emotion-Aware Drowsiness Detection System")
st.markdown(
    """
    **This system uses AI to monitor your eye state in real-time.**
    If drowsiness is detected, an alert sound will play to prevent road accidents.
    """
)

# ---------------------------
# ⚙️ Helper Functions
# ---------------------------
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def predict_eye_state(frame):
    img = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return "Closed" if prediction > 0.5 else "Open"

def play_alert():
    # Base64 encoded alert sound (short beep)
    b64_sound = base64.b64encode(open("alert.mp3", "rb").read()).decode()
    sound_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64_sound}" type="audio/mp3">
    </audio>
    """
    st.markdown(sound_html, unsafe_allow_html=True)

# ---------------------------
# 🎥 Real-time Webcam Detection
# ---------------------------
run = st.toggle("Start Camera")

FRAME_WINDOW = st.image([])
status_text = st.empty()
alert_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    st.success("✅ Camera is ON. Monitoring your eyes...")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to access camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        eye_state = predict_eye_state(frame_rgb)

        if eye_state == "Closed":
            cv2.putText(frame_rgb, "Drowsiness Detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            status_text.markdown("### ⚠️ **Drowsiness Detected! Please Stay Alert!**")
            play_alert()
        else:
            cv2.putText(frame_rgb, "Eyes Open", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            status_text.markdown("### ✅ **Eyes Open — You’re alert!**")

        FRAME_WINDOW.image(frame_rgb)
        time.sleep(1.5)  # Adjust detection speed

    cap.release()
    st.warning("🛑 Camera stopped.")
else:
    st.info("👆 Toggle the switch to start real-time drowsiness detection.")

# ---------------------------
# 📋 Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built by <b>Tamanna Vaikkath</b> | Emotion-Aware Driver Safety System</p>",
    unsafe_allow_html=True,
)
