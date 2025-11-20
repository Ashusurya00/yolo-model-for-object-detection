
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# ---------------- Centroid Tracker (lightweight) ----------------
class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        # next object ID to assign
        self.next_object_id = 0
        # dict of object_id -> centroid
        self.objects = dict()
        # number of consecutive frames an object has disappeared
        self.disappeared = dict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # rects: list of bounding boxes (x1,y1,x2,y2)
        if len(rects) == 0:
            # increment disappeared counters
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        # compute input centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        # if no existing objects, register all
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # build object IDs and centroids arrays
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # compute distance matrix between object_centroids and input_centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)

            # find smallest value's row and column indices sorted
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # compute unused rows and cols
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # if number of object centroids >= input centroids
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

# ---------------- Utility functions ----------------

def draw_annotations(frame, detections, tracker=None, class_names=None):
    # detections: list of (x1,y1,x2,y2,conf,class_id,label)
    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        label = str(cls) if class_names is None else f"{class_names[int(cls)]}"
        # box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # label
        t = f"{label} {conf:.2f}"
        cv2.putText(frame, t, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # draw tracked IDs if tracker provided
    if tracker is not None:
        for oid, centroid in tracker.items():
            cv2.putText(frame, f"ID {oid}", (int(centroid[0]) - 10, int(centroid[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (0,0,255), -1)
    return frame

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="YOLOv8 Realtime + Tracking", layout="wide")
st.title("YOLOv8 Real-time Object Detection + Tracking — Streamlit")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Model", options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
    conf_thres = st.slider("Confidence threshold", 0.1, 1.0, 0.35, 0.01)
    iou_thres = st.slider("NMS IoU", 0.1, 1.0, 0.45, 0.01)
    enable_tracking = st.checkbox("Enable tracking (CentroidTracker)", value=True)
    max_disappeared = st.number_input("Max disappeared frames (tracking)", min_value=1, max_value=500, value=50)
    max_distance = st.number_input("Max centroid distance (px)", min_value=5, max_value=500, value=60)
    source_type = st.radio("Source", ["Webcam", "Upload Video", "Sample Video"]) 
    if source_type == "Sample Video":
        sample_path = st.text_input("Sample video path (local)", value="sample.mp4")
    start_button = st.button("Start")
    stop_button = st.button("Stop")

# load model (lazy load for faster UI)
@st.cache_resource
def load_model(model_name):
    try:
        model = YOLO(model_name)
    except Exception as e:
        st.error(f"Failed to load model {model_name}: {e}")
        raise
    return model

model = load_model(model_choice)

# class names from model
class_names = model.names if hasattr(model, 'names') else None

# video capture setup
video_file_buffer = None
if source_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        tfile = uploaded_file.read()
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(tfile)
        tmp.flush()
        video_file_buffer = tmp.name

# placeholder for output
frame_placeholder = st.empty()
info_placeholder = st.empty()

# control variables
running = False

if start_button:
    running = True

if stop_button:
    running = False

# main loop
if running:
    # choose source
    if source_type == "Webcam":
        src = 0
    elif source_type == "Upload Video":
        if video_file_buffer is None:
            st.warning("Please upload a video first.")
            st.stop()
        src = video_file_buffer
    else:
        src = sample_path

    cap = cv2.VideoCapture(src)
    tracker = CentroidTracker(max_disappeared=int(max_disappeared), max_distance=int(max_distance)) if enable_tracking else None

    prev_time = time.time()
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # convert BGR to RGB for model
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # run detection
            # Using ultralytics YOLO predict on a numpy array returns results
            results = model(rgb, conf=conf_thres, iou=iou_thres, verbose=False)
            # results is list-like; take first
            res = results[0]

            detections = []
            rects_for_tracking = []
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes.data.tolist():
                    # box format: [x1,y1,x2,y2,confidence,class]
                    x1, y1, x2, y2, conf, cls = box
                    detections.append((x1, y1, x2, y2, float(conf), int(cls)))
                    rects_for_tracking.append((x1, y1, x2, y2))

            # update tracker
            tracked_objects = None
            if tracker is not None:
                tracked_objects = tracker.update(rects_for_tracking)

            # draw annotations
            annotated = draw_annotations(frame.copy(), detections, tracker=tracked_objects, class_names=class_names)

            # show frame
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels="RGB")

            # fps info
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            info_placeholder.markdown(f"**Frame:** {frame_count} &nbsp;&nbsp; **Detections:** {len(detections)} &nbsp;&nbsp; **FPS:** {fps:.2f}")

            # small sleep to yield control (helps Streamlit)
            if source_type == "Webcam":
                time.sleep(0.02)

            # handle stop button by checking session state - quick hack
            if st.session_state.get('stop_pressed'):
                break

    except Exception as e:
        st.error(f"Error during processing: {e}")
    finally:
        cap.release()

else:
    st.info("Press Start in the sidebar to begin detection (Webcam or Upload a video).")


