# 😐 Real-Time Face Emotion Detection using Deep Learning

A real-time face emotion detection system built using **Python, OpenCV, and TensorFlow**.
The model detects human facial emotions through a webcam and displays stable predictions.

---

## 🚀 Features
- Real-time webcam-based face emotion detection
- CNN-based facial expression classification
- Emotion smoothing (frame skipping + majority voting)
- Optimized for Windows (DirectShow backend)
- Beginner-friendly & well-documented

---

## 🛠️ Tech Stack
- Python 3.10
- TensorFlow (CPU)
- OpenCV
- NumPy
- SciPy

---

## 📂 Project Structure
Face-Emotion-Detection/
├── face_emotion.py
├── train_face_emotion.py
├── requirements.txt
└── README.md


---

## ▶️ How to Run

```bash
# Create virtual environment
python -m venv mental_env
mental_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_face_emotion.py

# Run real-time emotion detection
python face_emotion.py

