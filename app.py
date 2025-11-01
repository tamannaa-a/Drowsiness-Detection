# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from utils import eye_aspect_ratio, play_beep
import time

st.set_page_config(page_title="Drowsiness Detection (Eye EAR)", layout="wide")

st.title("🚗 Emotion-aware Driving: Eye State (Open/Closed) — Streamlit (Local Webcam)")
st.caption("Uses MediaPipe FaceMesh + EAR. Run this locally to allow webcam access (Streamlit can't access local hardware when hosted on Render without extra signalling).")

start = st.button("Start Webcam")
stop = st.button("Stop Webcam")

FRAME_WINDOW = st.image([])

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Eye landmark indices for MediaPipe FaceMesh (right and left eye)
# We'll use a subset approximating 6 relevant points per eye for EAR calculation.
# These indices are from MediaPipe's 468-point mesh.
# Right eye (as subject's right): use points approximating outer/inner and vertical pairings
RIGHT_EYE_IDX = [33, 160, 158, 133, 144, 153]   # approximate 6 points
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]  # approximate 6 points

EAR_THRESH = 0.22         # tuneable (lower means stricter closed detection)
CONSEC_FRAMES = 15        # how many consecutive frames below threshold before alarm

cap = None
if "running" not in st.session_state:
    st.session_state.running = False
if "counter" not in st.session_state:
    st.session_state.counter = 0
if "alarm_on" not in st.session_state:
    st.session_state.alarm_on = False

if start:
    st.session_state.running = True
    st.session_state.counter = 0
    st.session_state.alarm_on = False

if stop:
    st.session_state.running = False
    if cap is not None:
        try:
            cap.release()
        except:
            pass

if st.session_state.running:
    # open webcam
    if cap is None:
        cap = cv2.VideoCapture(0)
        # try second camera if 0 fails
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Could not open webcam. Make sure your webcam is connected and allowed.")
    else:
        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Empty frame. Retrying...")
                    time.sleep(0.1)
                    continue

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(img_rgb)

                status_text = "No face"
                ear_val = None

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    h, w, _ = frame.shape

                    def landmark_coords(idx):
                        lm = face_landmarks.landmark[idx]
                        return np.array([lm.x * w, lm.y * h])

                    # gather points
                    r_pts = [landmark_coords(i) for i in RIGHT_EYE_IDX]
                    l_pts = [landmark_coords(i) for i in LEFT_EYE_IDX]

                    # compute EAR for both eyes
                    ear_r = eye_aspect_ratio(r_pts)
                    ear_l = eye_aspect_ratio(l_pts)
                    ear_val = (ear_r + ear_l) / 2.0

                    # draw eye contours
                    for p in r_pts:
                        cv2.circle(frame, tuple(p.astype(int)), 2, (0,255,0), -1)
                    for p in l_pts:
                        cv2.circle(frame, tuple(p.astype(int)), 2, (0,255,0), -1)

                    # decide state
                    if ear_val < EAR_THRESH:
                        st.session_state.counter += 1
                    else:
                        st.session_state.counter = 0
                        st.session_state.alarm_on = False

                    if st.session_state.counter >= CONSEC_FRAMES:
                        status_text = "DROWSY / Eyes CLOSED"
                        # trigger beep if not already on
                        if not st.session_state.alarm_on:
                            st.session_state.alarm_on = True
                            play_beep(frequency=1000, duration_ms=350, volume=0.35)
                    else:
                        status_text = "Eyes OPEN" if ear_val is not None else "No face"

                    # overlay text
                    cv2.putText(frame, f"EAR: {ear_val:.3f}" if ear_val is not None else "EAR: --",
                                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                    cv2.putText(frame, status_text, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0,0,255) if "DROWSY" in status_text else (0,255,0), 2)
                else:
                    st.session_state.counter = 0
                    st.session_state.alarm_on = False

                # show image
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # small sleep to reduce CPU usage
                time.sleep(0.02)

        except Exception as e:
            st.error(f"Streaming loop exited: {e}")
        finally:
            if cap:
                cap.release()
            st.session_state.running = False
else:
    st.info("Press **Start Webcam** to begin real-time detection (this uses your local webcam).")
    st.markdown(
        """
        **Notes**
        - This app must be run *locally* so it can access your webcam: `streamlit run streamlit_app.py`.
        - If you want to deploy to cloud (Render) and still use a browser webcam, see the README (or ask and I'll provide a streamlit-webrtc version).
        """
    )
