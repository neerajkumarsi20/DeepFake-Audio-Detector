import streamlit as st
import numpy as np
import librosa as lb
import tensorflow as tf

st.set_page_config(page_title="Deepfake Audio Detector", layout="centered")
st.title("🎙️ Deepfake Audio Detection")
st.write("Upload a `.wav` audio file to check if it's Real or Fake.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_deepfake_detector.keras")

def preprocess_audio(file, sr=22050, n_mels=64, duration=1.5):
    y, sr = lb.load(file, sr=sr)
    sample_len = int(sr * duration)
    if len(y) < sample_len:
        y = np.pad(y, (0, sample_len - len(y)), 'constant')
    else:
        y = y[:sample_len]
    
    # Mel spectrogram
    mels = lb.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mels_db = lb.power_to_db(mels, ref=np.max)
    mels_db = np.abs(mels_db) / 80.0  # Normalize like in training
    
    return mels_db.reshape(1, mels_db.shape[0], mels_db.shape[1], 1)

if uploaded_file:
    st.audio(uploaded_file)

    model = load_model()
    features = preprocess_audio(uploaded_file)

    pred = model.predict(features)[0][0]
    label = "🧠 Fake" if pred > 0.5 else "✅ Real"
    st.markdown(f"### Prediction: **{label}**")
    st.caption(f"Confidence: {pred:.2f}")
