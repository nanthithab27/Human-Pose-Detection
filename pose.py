import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from io import BytesIO

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Streamlit setup
st.set_page_config(page_title="Pose Estimation with MediaPipe", layout="wide")

# Title of the web app
st.title("Pose Estimation with MediaPipe and Streamlit")
st.markdown("""
    Upload an image or video for pose estimation using MediaPipe Pose. 
    The app detects key body landmarks and visualizes them.
""")

# Function to process and display pose landmarks for image input
def process_image(image):
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            st.write("No pose detected.")
            return None

        # Draw pose landmarks on the image
        annotated_image = image.copy()
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (192, 192, 192)  # Background color
        annotated_image = np.where(condition, annotated_image, bg_image)
        
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        return annotated_image

# Function to process and display pose landmarks for video input
def process_video(video_file):
    cap = cv2.VideoCapture(video_file)
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        stframe = st.empty()  # Placeholder for video frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Draw pose landmarks on the image
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_rgb,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # Display video frame in Streamlit
            stframe.image(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        cap.release()

# Upload Image
image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    image = np.array(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, 1)
    
    annotated_image = process_image(image)
    
    if annotated_image is not None:
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Pose Estimation on Image")

# Upload Video
video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if video_file is not None:
    # Convert video file to a temporary location for OpenCV to read
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())
    
    process_video(temp_video_path)
