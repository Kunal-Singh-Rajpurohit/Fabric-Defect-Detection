from PIL import Image
import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os

MODEL_DIR = './runs/detect/train/weights/best.pt'
EXTRACT_DIR = './extracted_frames'

def main():
    st.title("🧵 Real-time Fabric Defect Detection")
    st.write("Upload an image or video to detect fabric defects using a YOLO model.")

    # Load YOLO model
    model = YOLO(MODEL_DIR)

    # File uploader
    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])

    # Frame extraction toggle
    extract_frames_toggle = st.checkbox("Extract frames for training", value=False)

    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            inference_images(uploaded_file, model)
        elif uploaded_file.type.startswith('video'):
            inference_video(uploaded_file, model, extract=extract_frames_toggle)

def inference_images(uploaded_file, model):
    image = Image.open(uploaded_file)
    prediction = model.predict(image)
    plotted = prediction[0].plot()[:, :, ::-1]
    boxes = prediction[0].boxes

    if len(boxes) == 0:
        st.markdown("**No Detection**")
    st.image(plotted, caption="Detected Image", use_container_width=True)

def inference_video(uploaded_file, model, extract=False, frame_skip=5):
    # Save video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    frame_count = 0
    extracted_count = 0

    if extract:
        os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Layout for original and detection video side by side
    col1, col2 = st.columns(2)
    col1.markdown("### 🎥 Original Video")
    col2.markdown("### 🧠 Detected Output")
    raw_frame_display = col1.empty()
    detection_frame_display = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Original frame
        raw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frame_display.image(raw_frame, channels="RGB", use_container_width=True)

        # YOLO detection
        results = model.predict(frame)[0]
        frame_with_boxes = results.plot()
        detection_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        detection_frame_display.image(detection_frame, channels="RGB", use_container_width=True)

        # Optional: save frame
        if extract and frame_count % frame_skip == 0:
            frame_filename = os.path.join(EXTRACT_DIR, f"frame_{extracted_count:05}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    os.remove(tfile.name)

    if extract:
        st.success(f"✅ Extracted {extracted_count} frames to `{EXTRACT_DIR}`")

if __name__ == '__main__':
    main()
