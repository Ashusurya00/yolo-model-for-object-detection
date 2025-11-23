# ============================================================
# ENVIRONMENT FIXES FOR STREAMLIT CLOUD
# ============================================================
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ":0"

# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

# ============================================================
# STREAMLIT UI SETTINGS
# ============================================================
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🚀 YOLOv8 Object Detection (Video Upload Only)")


# ============================================================
# MODEL SELECTION & PARAMETERS
# ============================================================
with st.sidebar:
    st.header("⚙ Settings")

    model_name = st.selectbox("Choose YOLO Model", ["yolov8n.pt"], index=0)
    conf = st.slider("Confidence Level", min_value=0.1, max_value=1.0, value=0.5)
    start_btn = st.button("▶ Start Detection")


# ============================================================
# LOAD YOLO MODEL
# ============================================================
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model = load_model(model_name)


# ============================================================
# VIDEO UPLOAD
# ============================================================
uploaded_video = st.file_uploader("Upload a video file",
                                  type=["mp4", "avi", "mov", "mkv"])

video_path = None

if uploaded_video:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_video.read())
    video_path = temp_file.name


# PLACEHOLDERS FOR OUTPUT
frame_display = st.empty()
info_display = st.empty()


# ============================================================
# DETECTION FUNCTION
# ============================================================
def run_detection(path):

    cap = cv2.VideoCapture(path)
    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO
        results = model(rgb, conf=conf, verbose=False)[0]

        # Draw bounding boxes
        if results.boxes is not None:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = box

                cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)

                label = f"{model.names[int(cls)]}: {score:.2f}"
                cv2.putText(rgb, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display frame
        frame_display.image(rgb, channels="RGB")

        # FPS Calculation
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        info_display.write(f"🎯 Frame: {frame_count} | ⚡ FPS: {fps:.2f}")


    cap.release()


# ============================================================
# START BUTTON LOGIC
# ============================================================
if start_btn:
    if video_path is None:
        st.error("⚠ Please upload a video first.")
    else:
        st.success("🎬 Detection Started…")
        run_detection(video_path)
