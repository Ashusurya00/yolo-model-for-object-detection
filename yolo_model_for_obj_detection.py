import streamlit as st
from ultralytics import YOLO
import supervision as sv
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🚀 YOLOv8 Object Detection (No OpenCV / Cloud Compatible)")

# Sidebar UI
with st.sidebar:
    st.header("⚙ Settings")
    model_name = st.selectbox("Choose YOLO Model", ["yolov8n.pt"])
    conf = st.slider("Confidence", 0.1, 1.0, 0.5)
    start = st.button("▶ Detect Objects")

# Load model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model = load_model(model_name)

# Upload video
uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "mkv", "avi"])
placeholder = st.empty()

def process_video(video_path):
    video = sv.VideoSink(target_path=None)
    box_annotator = sv.BoxAnnotator()

    for frame in sv.get_video_frames_generator(video_path):
        results = model(frame, conf=conf)[0]
        detections = sv.Detections.from_yolov8(results)
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        placeholder.image(annotated_frame, channels="RGB")

if start:
    if not uploaded:
        st.error("Upload a video first!")
    else:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(uploaded.read())
        st.success("Processing video... please wait")
        process_video(temp.name)
