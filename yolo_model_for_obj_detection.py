# ============================================================
#  FIXES FOR STREAMLIT CLOUD DEPLOYMENT
# ============================================================
import os
os.environ["UV_CACHE_DIR"] = "/tmp"   # avoids corrupted YOLO weights
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"

# ============================================================
#  IMPORTS
# ============================================================
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

# ============================================================
#  STREAMLIT SETTINGS
# ============================================================
st.set_page_config(page_title="YOLO Real-time Detection", layout="wide")
st.title("🚀 YOLO Real-Time Object Detection (Streamlit Cloud Ready)")

with st.sidebar:
    st.header("⚙ Settings")

    # ONLY YOLOv8n – reduces errors & huge weight downloads
    model_choice = st.selectbox("Select Model", ["yolov8n.pt"], index=0)

    conf = st.slider("Confidence", 0.1, 1.0, 0.5)
    source_type = st.radio("Input Source", ["Webcam", "Upload Video"])

    start_btn = st.button("▶ Start Detection")
    stop_btn = st.button("⛔ Stop Detection")


# ============================================================
#  LOAD MODEL  — DO NOT CACHE (prevents pickle errors)
# ============================================================
def load_model(m):
    return YOLO(m)

model = load_model(model_choice)


# ============================================================
#  VIDEO SOURCE HANDLING
# ============================================================
video_path = None

if source_type == "Upload Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(uploaded.read())
        video_path = temp.name


# Output placeholders
frame_area = st.empty()
info_area = st.empty()


# ============================================================
#  REAL-TIME DETECTION LOOP
# ============================================================
def run_detection(source):

    cap = cv2.VideoCapture(source)
    prev_time = time.time()
    frame_count = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO prediction
        results = model(rgb, conf=conf, verbose=False)[0]

        detections = []

        if results.boxes is not None:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = box
                detections.append((x1, y1, x2, y2, score, cls))

                # Draw boxes
                cv2.rectangle(rgb, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)

                label = f"{model.names[int(cls)]} {score:.2f}"
                cv2.putText(rgb, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show output frame
        frame_area.image(rgb, channels="RGB")

        # FPS calculation
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        info_area.markdown(
            f"📌 **Frame:** {frame_count} | 🎯 **Detections:** {len(detections)} | ⚡ **FPS:** {fps:.2f}"
        )

        # Small delay to prevent Streamlit freezing
        time.sleep(0.02)

    cap.release()


# ============================================================
#  START BUTTON LOGIC
# ============================================================
if start_btn:

    st.success("🚀 Detection Started…")

    if source_type == "Webcam":
        run_detection(0)

    elif source_type == "Upload Video":
        if video_path is None:
            st.error("Upload a video to start detection.")
        else:
            run_detection(video_path)

elif stop_btn:
    st.warning("⛔ Detection Stopped.")
