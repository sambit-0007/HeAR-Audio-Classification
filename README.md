#### 🩺 Respiratory Disease Classification API 
This project uses Google's HeAR (Health Acoustics Representations) model to classify respiratory sounds into different disease categories using audio embeddings. 
#### Overview 

This service takes short audio clips (2 seconds, .wav format) of respiratory sounds and classifies them into one of the following categories: 

Asthma 
Picture COPD 
Pneumonia 
Bronchial 
Healthy 

The system uses Google's HeAR model to generate embeddings from the audio data, which are then fed into a Random Forest classifier to predict the respiratory condition. 

#### Technical Architecture 

Frontend: Streamlit web interface for easy file upload and classification 
Backend API: Flask REST API for programmatic access 
Models: 
  Google HeAR model for audio embedding extraction 
  Random Forest classifier trained on respiratory sound dataset 

#### Getting Started 
##### Prerequisites
        Python 3.8+ 
        PyTorch 
        Google's HeAR model 
        Required Python packages  
        
#### Installation 

Clone the repository: 
    git clone https://github.com/sambit-0007/HeAR-Audio-Classification  
    cd HeAR-Audio-Classification 

Install the required packages: 
    pip install -r requirements.txt 

Running the API 
Start the Flask API: 
    python app.py 

Running the Streamlit Interface 
Start the Streamlit app: 
    streamlit run streamlit_app.py


#### API Usage 

    POST /predict 

    Endpoint for classifying respiratory audio. 

    Request: 
      Method: POST 
        Content-Type: multipart/form-data 


Body: Form data with a "file" field containing a .wav audio file (2 seconds) 


Response: 
  200 OK: JSON object with predicted class 
json 

{ 
"predicted_class": "asthma" 
} 

400 Bad Request: Error if no file or wrong format 
json 

{ 
"error": "No file provided" 
} 

500 Internal Server Error: Other errors 
json 
{ 
"error": "Error message details" 
} 

#### Test Dataset
To test whether the model is working or not please use the images from the test_dataset folder. 

#### Known Limitations 

The model requires exactly 2-second audio clips. Shorter clips will be padded, and longer clips will be truncated. 
Performance on bronchial conditions is currently limited due to dataset imbalance. Image 62, Picture Audio must be in .wav format with a sample rate of 16kHz. 

#### Future Improvements 
Enhance model performance with additional training data Image 64, Picture Add confidence scores to predictions 
Support for longer audio clips with sliding window analysis Image 66, Picture Add model versioning and A/B testing 

#### Dataset 

This project uses the "Asthma Detection Dataset Version 2" which includes labeled respiratory sounds for various conditions. The dataset was split 80/20 for training and testing. 
 
 
