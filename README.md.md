#  Driver Drowsiness Detection System

A real-time driver drowsiness detection system built using OpenCV,
MediaPipe, and TensorFlow (MobileNetV2). The system uses live
webcam footage to detect eye state (open/closed), track facial
landmarks, and trigger alerts when drowsiness is detected.

##  Features

-   Real-time webcam detection
-   Eye-state classification: Open vs Closed
-   Facial landmark tracking using MediaPipe Face Mesh
-   Deep learning model trained on MRL Eye Dataset
-   Alerts user with alarm sound if drowsiness is detected
-   Lightweight & optimized for real-time performance

##  Technology Stack

-   Python
-   OpenCV
-   MediaPipe
-   TensorFlow / Keras
-   MobileNetV2
-   NumPy, Pandas
-   Jupyter Notebook

## 📁 Project Structure

    driver-drowsiness-detection/
    ├── app.py
    ├── app.ipynb
    ├── mobilenet_model.h5
    ├── requirements.txt
    └── README.md

##  How It Works

### 1️ Face & Eye Detection

Uses "MediaPipe Face Mesh"to detect eye contours & landmarks.

### 2️ Eye Classification

MobileNetV2 CNN model predicts: - 0 → Closed - 1 → Open

### 3️ Drowsiness Logic

If eyes are closed for a threshold duration, an alert triggers.

##  Installation & Usage

``` bash
git clone https://github.com/Akshat6708/driver-drowsiness-detection
cd driver-drowsiness-detection
pip install -r requirements.txt
python app.py
```

##  Model Details

-   Model: MobileNetV2
-   Dataset: MRL Eye Dataset
-   Training notebook: `app.ipynb`

##  Future Improvements

-   Yawn detection
-   Head pose estimation
-   TensorFlow Lite build
-   IoT integration (Raspberry Pi)

##  Author

Akshat Patidar\
GitHub: https://github.com/Akshat6708\
LinkedIn: https://www.linkedin.com/in/akshat-patidar-a3906a250


