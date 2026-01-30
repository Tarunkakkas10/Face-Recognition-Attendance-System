import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from collections import deque

# ---------------- CONFIG ----------------
CONFIDENCE_THRESHOLD = 85      # stricter
MARGIN_THRESHOLD = 20          # stronger unknown detection
IMG_SIZE = 100
MODEL_PATH = "model/face_recognition_model.h5"
DATASET_PATH = "dataset"

SMOOTHING_FRAMES = 7           # number of frames to vote
# ----------------------------------------

# Load model
model = load_model(MODEL_PATH)
labels = sorted(os.listdir(DATASET_PATH))

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Prediction buffer (for smoothing)
prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        probs = model.predict(face, verbose=0)[0] * 100
        top1 = np.max(probs)
        top2 = np.sort(probs)[-2]
        margin = top1 - top2

        label_index = np.argmax(probs)
        predicted_label = labels[label_index]

        # Raw decision
        if top1 >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
            prediction_buffer.append(predicted_label)
        else:
            prediction_buffer.append("Unknown")

        # Majority voting
        final_label = max(set(prediction_buffer), key=prediction_buffer.count)

        color = (0, 255, 0) if final_label != "Unknown" else (0, 0, 255)
        text = f"{final_label}"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

    cv2.imshow("Face Recognition (Stable)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
