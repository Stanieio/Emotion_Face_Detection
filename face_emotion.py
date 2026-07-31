import cv2
import sys
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque, defaultdict

# ---------------- Config ----------------
MODEL_PATH = "face_emotion_model.h5"
EMOTION_LABELS = ['Angry', 'Happy', 'Sad', 'Neutral']
PREDICT_EVERY_N_FRAMES = 5
HISTORY_LEN = 15
MIN_FACE_SIZE = 80
CONFIDENCE_THRESHOLD = 0.5
# -----------------------------------------

def load_resources():
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"ERROR: could not load model '{MODEL_PATH}': {e}")
        sys.exit(1)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"ERROR: could not load cascade at {cascade_path}")
        sys.exit(1)

    return model, face_cascade


def match_face_id(x, y, w, h, tracked, max_dist=60):
    """Cheap centroid-based tracking so each face keeps its own emotion history."""
    cx, cy = x + w / 2, y + h / 2
    for fid, (tx, ty) in tracked.items():
        if abs(cx - tx) < max_dist and abs(cy - ty) < max_dist:
            tracked[fid] = (cx, cy)
            return fid
    new_id = max(tracked.keys(), default=-1) + 1
    tracked[new_id] = (cx, cy)
    return new_id


def main():
    model, face_cascade = load_resources()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW is Windows-only; drop it on Mac/Linux
    if not cap.isOpened():
        print("ERROR: Camera could not be opened")
        sys.exit(1)

    print("Press Q to quit")

    frame_idx = 0
    tracked_positions = {}                       # face_id -> (cx, cy)
    histories = defaultdict(lambda: deque(maxlen=HISTORY_LEN))  # face_id -> emotion deque
    labels = defaultdict(lambda: "Detecting...")  # face_id -> current stable label

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # improves detection under uneven lighting

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
            )

            do_predict = (frame_idx % PREDICT_EVERY_N_FRAMES == 0)

            for (x, y, w, h) in faces:
                face_id = match_face_id(x, y, w, h, tracked_positions)

                if do_predict:
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (48, 48))
                    face_roi = face_roi.astype("float32") / 255.0
                    face_roi = face_roi.reshape(1, 48, 48, 1)

                    prediction = model.predict(face_roi, verbose=0)[0]
                    best_idx = int(np.argmax(prediction))
                    confidence = float(prediction[best_idx])

                    if confidence >= CONFIDENCE_THRESHOLD:
                        histories[face_id].append(EMOTION_LABELS[best_idx])

                    if histories[face_id]:
                        h_deque = histories[face_id]
                        labels[face_id] = max(set(h_deque), key=h_deque.count)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame, labels[face_id], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                )

            cv2.imshow("Face Emotion Detection (Smooth)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()