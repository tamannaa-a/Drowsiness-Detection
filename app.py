import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

st.title("Fatigue / Eye State Detection")
st.write("Detect whether eyes are open or closed in real-time.")

# Load model
model = load_model("fatigue_model.h5")

option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Upload image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0][0]
        st.success("Eyes OPEN 👀" if prediction>=0.5 else "Eyes CLOSED 😴")

# Webcam
if option == "Use Webcam":
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        processed = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224,224))
        processed = img_to_array(processed)/255.0
        processed = np.expand_dims(processed, axis=0)
        prediction = model.predict(processed)[0][0]
        label = "Eyes OPEN 👀" if prediction>=0.5 else "Eyes CLOSED 😴"
        cv2.putText(frame, label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
