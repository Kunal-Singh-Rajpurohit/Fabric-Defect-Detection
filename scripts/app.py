from PIL import Image
import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os

MODEL_DIR = './runs/detect/train/weights/best.pt'

def main():
    st.title("Real-time Fabric Defect Detection")
    st.write("The aim of this project is to develop an efficient computer vision model capable of real-time Fabric Defect detection.")

    # Load model
    model = YOLO(MODEL_DIR)

    # Upload file
    uploaded_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])

    if uploaded_file:
        if uploaded_file.type.startswith('image'):
            inference_images(uploaded_file, model)
        elif uploaded_file.type.startswith('video'):
            inference_video(uploaded_file, model)

def inference_images(uploaded_file, model):
    image = Image.open(uploaded_file)
    prediction = model.predict(image)
    plotted = prediction[0].plot()[:, :, ::-1]  # Convert BGR to RGB
    boxes = prediction[0].boxes

    if len(boxes) == 0:
        st.markdown("**No Detection**")
    st.image(plotted, caption="Detected Image", width=600)

def inference_video(uploaded_file, model):
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)[0]
        frame_with_boxes = results.plot()

        # Display the frame in Streamlit
        stframe.image(frame_with_boxes[:, :, ::-1], channels="RGB", use_column_width=True)

    cap.release()
    os.remove(tfile.name)

if __name__ == '__main__':
    main()
