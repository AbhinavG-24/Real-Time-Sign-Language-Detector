import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import json

MODEL_PATH = "gesture_model.h5"
LABEL_PATH = "label_map.json"

# Load model and label mapping
model = keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, "r") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    predicted_label = ""

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Normalize landmarks relative to wrist
        base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        landmarks = np.array(landmarks).reshape(1, -1)

        prediction = model.predict(landmarks, verbose=0)
        predicted_label = inv_label_map[np.argmax(prediction)]

    cv2.putText(frame, f"Prediction: {predicted_label}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),3)
    cv2.imshow("Sign Language Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
