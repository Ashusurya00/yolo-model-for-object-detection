import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import time


# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="YOLO Realtime Detection", layout="wide")
st.title("🚀 YOLO Real-Time Object Detection (Streamlit Cloud Compatible)")

with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Select YOLO Model",
        ["yolov8n.pt", "yolov8s.pt"],
        index=0
    )

    conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

    source_type = st.radio("Input Source", ["Webcam", "Upload Video"])

    start_btn = st.button("▶ Start Detection")
    stop_btn = st.button("⛔ Stop Detection")


# -------------------- Load YOLO Model --------------------
@st.cache_resource
def load_model(m):
    return YOLO(m)

model = load_model(model_choice)


# -------------------- Video Source Logic --------------------
video_path = None

if source_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(uploaded_file.read())
        video_path = temp.name


frame_area = st.empty()
info_area = st.empty()


# -------------------- Run Realtime Detection --------------------
def run_detection(video_src):
    cap = cv2.VideoCapture(video_src)

    prev_time = time.time()
    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO Prediction
        results = model(rgb_frame, conf=conf, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, score, cls_id = box
                detections.append((x1, y1, x2, y2, score, cls_id))

                # Draw rectangle
                cv2.rectangle(rgb_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)

                label = f"{model.names[int(cls_id)]} {score:.2f}"
                cv2.putText(rgb_frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display frame
        frame_area.image(rgb_frame, channels="RGB")

        # FPS calculations
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        info_area.markdown(
            f"**Frames:** {frame_count} | **Detections:** {len(detections)} | **FPS:** {fps:.2f}"
        )

        # Yield for Streamlit
        time.sleep(0.02)


    cap.release()


# -------------------- Start Button Logic --------------------
if start_btn:
    st.success("Detection started… ⏳")

    if source_type == "Webcam":
        run_detection(0)
    else:
        if video_path is None:
            st.error("Upload a video first!")
        else:
            run_detection(video_path)

elif stop_btn:
    st.warning("Detection stopped.")
