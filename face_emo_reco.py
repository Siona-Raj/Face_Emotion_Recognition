import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import os

# Store last 10 predictions for smoothing
recent_predictions = deque(maxlen=10)

# =========================
# Choose classifier: "haar" or "lbp"
classifier_choice = "haar"   # change to "lbp" if you want
# =========================

if classifier_choice == "haar":
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
else:
    # expects lbpcascade_frontalface.xml in project folder
    cascade_path = os.path.join(os.getcwd(), "lbpcascade_frontalface.xml")

face_cascade = cv2.CascadeClassifier(cascade_path)

# Check if cascade loaded
if face_cascade.empty():
    st.error(f"Could not load classifier! Check the path: {cascade_path}")
    st.stop()

# Load pre-trained emotion model
emotion_model = load_model("emotion_model.h5", compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print("Model loaded. Emotions =", emotion_labels)
print(f"Using {classifier_choice.upper()} classifier")

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # Preprocess face for emotion detection
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64,64))   # match model input
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)     # (1, 64, 64)
        face = np.expand_dims(face, axis=-1)    # (1, 64, 64, 1)

        # Predict emotion
        prediction = emotion_model.predict(face, verbose=0)[0]
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        confidence = int(prediction[emotion_index] * 100)  # percentage

        # Add to recent predictions
        recent_predictions.append(emotion)
        stable_emotion = max(set(recent_predictions), key=recent_predictions.count)

        # Display result with confidence
        label = f"{stable_emotion} ({confidence}%)"
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Show video feed
    cv2.imshow('Face & Emotion Detection', frame)
    
    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
