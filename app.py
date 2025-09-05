import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# ============ App Setup ============
st.set_page_config(
    page_title="Face & Emotion Detection",
    layout="wide"
)

# Custom CSS for a modern, minimal look
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
        margin: auto;
    }
    h1, h2, h3 {
        text-align: center;
        color: #2E86C1;
    }
    .stRadio > label {
        font-weight: 600;
        font-size: 1.05rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Face & Emotion Detection")
st.markdown(
    "This application can detect human faces and classify their emotions using deep learning."
)

# ============ Load Emotion Model ============
emotion_model = load_model("emotion_model.h5", compile=False)
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# ============ Sidebar Options ============
classifier_choice = st.sidebar.selectbox(
    "Choose Face Detection Classifier:",
    ["Haar Cascade", "LBP Cascade"]
)

# Load appropriate classifier
if classifier_choice == "Haar Cascade":
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
else:
    face_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")

# ============ Helper ============
def detect_emotions(img):
    """Detect faces & emotions in an image"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (46, 204, 113), 2)
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64,64))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        prediction = emotion_model.predict(face, verbose=0)[0]
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        confidence = int(prediction[emotion_index] * 100)

        cv2.putText(img, f"{emotion} ({confidence}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (46, 204, 113), 2)

    return img


# ============ Input Mode ============
mode = st.radio(
    "Select Input Mode:",
    ["Webcam", "Upload Image", "Upload Video"],
    horizontal=True
)

# --- Webcam Mode ---
if mode == "Webcam":
    st.subheader("Webcam Capture")
    picture = st.camera_input("Take a picture")
    if picture:
        img = Image.open(picture)
        img = np.array(img.convert('RGB'))
        result = detect_emotions(img)
        st.image(result, caption="Detected Faces & Emotions", use_container_width=True)

# --- Image Upload ---
elif mode == "Upload Image":
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = np.array(img.convert('RGB'))
        result = detect_emotions(img)
        st.image(result, caption="Detected Faces & Emotions", use_container_width=True)

# --- Video Upload ---
elif mode == "Upload Video":
    st.subheader("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4","avi","mov"])
    if uploaded_video is not None:
        tfile = open("temp_video.mp4", "wb")
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture("temp_video.mp4")

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = detect_emotions(frame)
            stframe.image(frame, channels="RGB", use_container_width=True)
        cap.release()
