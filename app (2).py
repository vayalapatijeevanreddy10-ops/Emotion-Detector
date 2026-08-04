"""
Streamlit Emotion Detection App
================================
Loads the saved model/scaler/label_encoder + MediaPipe Face Landmarker,
and predicts emotion from either an uploaded image or a webcam snapshot.

Run with:
    streamlit run app.py

Expected folder structure:
    app.py
    face_landmarker.task
    models/
        emotion_model.keras
        scaler.pkl
        label_encoder.pkl
"""

import os
import numpy as np
import cv2
import joblib
import streamlit as st
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Emotion Detector", page_icon="🙂", layout="centered")

MODEL_PATH = "models/emotion_model.keras"
SCALER_PATH = "models/scaler.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"
LANDMARKER_PATH = "face_landmarker.task"


# ------------------------------------------------------------------
# Cached loaders — these run once and are reused across interactions
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_label_encoder():
    return joblib.load(LABEL_ENCODER_PATH)


@st.cache_resource
def load_face_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# ------------------------------------------------------------------
# Inference pipeline — same logic as Section 16 of the notebook
# ------------------------------------------------------------------
def predict_emotion(rgb_image, model, scaler, label_encoder, face_landmarker):
    """
    rgb_image: np.array, RGB, uint8
    Returns (predicted_emotion, confidence) or (None, None) if no face found.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = face_landmarker.detect(mp_image)

    if len(detection_result.face_landmarks) == 0:
        return None, None

    landmarks = detection_result.face_landmarks[0]
    landmark_vector = []
    for landmark in landmarks:
        landmark_vector.extend([landmark.x, landmark.y, landmark.z])
    landmark_vector = np.array(landmark_vector, dtype=np.float32).reshape(1, -1)

    scaled_vector = scaler.transform(landmark_vector)

    pred_probs = model.predict(scaled_vector, verbose=0)
    pred_class_idx = np.argmax(pred_probs)
    confidence = pred_probs[0][pred_class_idx]

    predicted_emotion = label_encoder.inverse_transform([pred_class_idx])[0]
    return predicted_emotion, float(confidence)


# ------------------------------------------------------------------
# Check required files exist before doing anything else
# ------------------------------------------------------------------
missing = [p for p in [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, LANDMARKER_PATH] if not os.path.exists(p)]
if missing:
    st.error(
        "Missing required file(s):\n\n"
        + "\n".join(f"- `{m}`" for m in missing)
        + "\n\nMake sure `app.py` sits next to `face_landmarker.task` and the `models/` folder."
    )
    st.stop()

model = load_model()
scaler = load_scaler()
label_encoder = load_label_encoder()
face_landmarker = load_face_landmarker()

EMOTION_EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🙂 Emotion Detector")
st.write("Upload a photo or use your webcam — the app will detect the face and predict the emotion.")

tab_upload, tab_webcam = st.tabs(["📁 Upload Image", "📷 Webcam"])

input_image = None  # will hold a PIL Image if we get one

with tab_upload:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

with tab_webcam:
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        input_image = Image.open(camera_file).convert("RGB")

# ------------------------------------------------------------------
# Run prediction if we have an image
# ------------------------------------------------------------------
if input_image is not None:
    rgb_image = np.array(input_image)

    with st.spinner("Detecting face and predicting emotion..."):
        predicted_emotion, confidence = predict_emotion(
            rgb_image, model, scaler, label_encoder, face_landmarker
        )

    st.image(rgb_image, caption="Input image", use_column_width=True)

    if predicted_emotion is None:
        st.warning("⚠ No face detected in this image. Try a clearer, front-facing photo.")
    else:
        emoji = EMOTION_EMOJI.get(predicted_emotion, "")
        st.success(f"**Predicted emotion:** {predicted_emotion.capitalize()} {emoji}")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.progress(min(max(confidence, 0.0), 1.0))
else:
    st.info("Upload an image or take a webcam photo to get started.")
