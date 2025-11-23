import streamlit as st
from ultralytics import YOLO
import supervision as sv
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🚀 YOLOv8 Object Detection (No OpenCV / Streamlit Cloud)")

# Sidebar
with st.sidebar:
    st.header("⚙ Settings")
    model_name = st.selectbox("YOLO Model", ["yolov8n.pt"])
    conf = st.slider("Confidence", 0.1, 1.0, 0.5)
    start = st.button("▶ Start Detection")

# Load model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model = load_model(model_name)

# Upload video
uploaded = st.file_uploader("Upload a video", type=["mp4", "mkv", "avi", "mov"])
frame_display = st.empty()


def process_video(video_path):
    box_annotator = sv.BoxAnnotator()

    for frame in sv.get_video_frames_generator(video_path):
        results = model(frame, conf=conf)[0]
        detections = sv.Detections.from_yolov8(results)

        # draw boxes
        annotated = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        frame_display.image(annotated, channels="RGB")


# Start processing
if start:
    if not uploaded:
        st.error("⚠ Upload a video first")
    else:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(uploaded.read())

        st.success("Processing video... please wait...")
        process_video(temp.name)
