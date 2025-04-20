# streamlit_app.py

import streamlit as st
import joblib
import tempfile
import librosa
import numpy as np
import torch
from python.data_processing.audio_utils import preprocess_audio
# from hear_utils import get_hear_embedding  # ensure correct import
import os

# Get the current working directory
current_path = os.getcwd()
print("Current working directory:", current_path)

# Load model
model = joblib.load(r"hear-master/model/classifier.pkl")


SAMPLE_RATE = 16000
EXPECTED_SAMPLES = SAMPLE_RATE * 2

def load_and_prepare_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < EXPECTED_SAMPLES:
        audio = np.pad(audio, (0, EXPECTED_SAMPLES - len(audio)))
    else:
        audio = audio[:EXPECTED_SAMPLES]
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

def get_hear_embedding(file_path):
    audio_tensor = load_and_prepare_audio(file_path)
    embedding = preprocess_audio(audio_tensor)  # [1, 1, 192, 128]
    return embedding.flatten().detach().numpy()  # [24576]

st.title("🩺 Disease Classification from Audio")
st.write("Upload a 2-second .wav clip to get the predicted class.")

# Upload audio
uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

if uploaded_file is not None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        embedding = get_hear_embedding(tmp_path)
        prediction = model.predict(embedding.reshape(1, -1))[0]
        st.success(f"🧠 Predicted Class: **{prediction}**")

    except Exception as e:
        st.error(f"Error: {e}")
