# Real-Time DeepFake Audio Detector 

![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Working-success)

A machine learning-powered application to detect deepfake (AI-generated) audio from real human speech using CNNs trained on mel spectrograms. Built with a Streamlit UI for real-time predictions

---

### 1. Clone the Repository

```bash
git clone https://github.com/neerajkumarsi20/DeepFake-Audio-Detector.git
cd DeepFake-Audio-Detector
```

### 2. Setup a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Finally Run the App

```bash
streamlit run app.py
```

To test use the sample image/video within data folder

## Highlighting Features

- 📊 Converts audio to mel spectrograms
- 🧠 Classifies audio as **Real** or **Fake** using a trained CNN
- 🎧 Supports real-time audio uploads and live predictions via a **Streamlit** interface