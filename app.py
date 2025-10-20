import streamlit as st
import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image

st.set_page_config(page_title="Fatigue / Eye State Detection", layout="centered")
st.title("Fatigue / Eye State Detection")
st.write("Detect whether eyes are open or closed in real-time.")

# -------------------- Load TFLite model --------------------
tflite_model_path = 'fatigue_model.tflite'
interpreter = Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------- Prediction Function --------------------
def predict_tflite(image):
    img = cv2.resize(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB), (224,224))
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return output

# -------------------- App UI --------------------
option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

# -------------------- Upload Image --------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")
        prediction = predict_tflite(image)
        if prediction >= 0.5:
            st.success("Eyes OPEN 👀")
        else:
            st.success("Eyes CLOSED 😴")

# -------------------- Webcam --------------------
if option == "Use Webcam":
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        prediction = predict_tflite(frame)
        label = "Eyes OPEN 👀" if prediction >= 0.5 else "Eyes CLOSED 😴"
        cv2.putText(frame, label, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
