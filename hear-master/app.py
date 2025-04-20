# flask_api.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
import tempfile
import librosa
import numpy as np
import os
from datetime import datetime
import torch
from python.data_processing.audio_utils import preprocess_audio

app = Flask(__name__)

# Load model
model = joblib.load(r"hear-master\model\classifier.pkl")



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

@app.route("/predict", methods=["POST"])
def predict():
    # Check if file exists in request
    if 'file' not in request.files:
        app.logger.error("No file provided in request")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    print(file)
    # Check if filename is empty
    if file.filename == '':
        app.logger.error("Empty filename provided")
        return jsonify({"error": "No file selected"}), 400
    
    # Check file format
    if not file.filename.endswith(".wav"):
        app.logger.error(f"Invalid file format: {file.filename}")
        return jsonify({"error": "Invalid file format. Please upload a .wav file."}), 400
    print(11)
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        app.logger.info(f"Processing file: {file.filename}")
        
        # Extract embedding and predict
        embedding = get_hear_embedding(temp_path)
        prediction = model.predict(embedding.reshape(1, -1))[0]
        print(12)
        # Get prediction probabilities for confidence
        probabilities = model.predict_proba(embedding.reshape(1, -1))[0]
        confidence = float(max(probabilities))
        
        app.logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
        
        return jsonify({
            "predicted_class": prediction,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        app.logger.error(f"Error processing file {file.filename}: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up temporary file
        if 'temp_path' in locals():
            try:
                os.remove(temp_path)
            except Exception as e:
                app.logger.warning(f"Failed to remove temporary file: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
