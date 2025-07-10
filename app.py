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

def tta_predict(file, model, sr=22050, n_mels=64, duration=1.5, stride=0.5):
    y, sr = lb.load(file, sr=sr)
    sample_len = int(sr * duration)
    stride_len = int(sr * stride)

    preds = []
    for start in range(0, len(y) - sample_len + 1, stride_len):
        chunk = y[start:start + sample_len]
        mels = lb.feature.melspectrogram(y=chunk, sr=sr, n_mels=n_mels)
        mels_db = lb.power_to_db(mels, ref=np.max)
        mels_db = np.abs(mels_db) / 80.0
        features = mels_db.reshape(1, n_mels, mels_db.shape[1], 1)
        pred = model.predict(features)[0][0]
        preds.append(pred)

    return np.mean(preds)

if uploaded_file:
    st.audio(uploaded_file)

    model = load_model()
    #features = preprocess_audio(uploaded_file)

    pred = tta_predict(uploaded_file, model)
    pred_rounded = round(pred, 2)
    label = "🧠 Fake" if pred_rounded >= 0.40 else "✅ Real"
    st.markdown(f"### Prediction: **{label}**")
    st.caption(f"Confidence: {pred:.2f}")
    