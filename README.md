# Face & Emotion Recognition

This project is a real-time face and emotion detection system using OpenCV and a deep learning model built with Keras/TensorFlow.

## Features
- Detects human faces in webcam video using Haar or LBP Cascade classifiers.
- Recognizes emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) on detected faces using a pre-trained model (`emotion_model.h5`).
- Smooths predictions over recent frames for stable results.
- Displays detected face(s) and predicted emotion(s) live on the video feed.

## Requirements
- Python 3.9 or higher
- OpenCV (`opencv-python`)
- TensorFlow
- Keras
- NumPy

## Setup & Usage
1. **Clone or download** this repository.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure the following files are present in the project folder:**
   - `face_emo_reco.py` (main script)
   - `emotion_model.h5` (pre-trained emotion recognition model)
   - `lbpcascade_frontalface.xml` (for LBP classifier, optional)
4. **Run the application:**
   ```bash
   python face_emo_reco.py
   ```
5. **Usage:**
   - The webcam will open and start detecting faces and emotions.
   - Press `q` to quit the application.

## Customization
- To switch between Haar and LBP face detection, change the `classifier_choice` variable in `face_emo_reco.py`.
- The emotion model expects grayscale face images of size 64x64.

## File Descriptions
- `face_emo_reco.py`: Main script for face and emotion detection.
- `emotion_model.h5`: Pre-trained Keras model for emotion recognition.
- `lbpcascade_frontalface.xml`: LBP face detector (optional, for LBP mode).
- `requirements.txt`: List of required Python packages.
- `CODE_EXPLANATION.md`: Detailed explanation of the code logic.

## License
For educational and research purposes.
