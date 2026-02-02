import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load trained model
model = load_model("face_emotion_model.h5")

# Emotion labels (must match training order)
emotion_labels = ['Angry', 'Happy', 'Sad', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam (DirectShow for Windows stability)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

print("Press Q to quit")

# -------- SMOOTHING SETTINGS --------
frame_count = 0
current_emotion = "Detecting..."
emotion_history = deque(maxlen=15)   # memory for voting
PREDICT_EVERY_N_FRAMES = 10           # frame skipping
MIN_FACE_SIZE = 80                   # ignore small faces
# -----------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Ignore very small faces (noise)
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)

        frame_count += 1

        # Predict only every N frames
        if frame_count % PREDICT_EVERY_N_FRAMES == 0:
            prediction = model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            emotion_history.append(emotion)

            # Majority voting for stability
            current_emotion = max(
                set(emotion_history),
                key=emotion_history.count
            )

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display stable emotion
        cv2.putText(
            frame,
            current_emotion,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Emotion Detection (Smooth)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
