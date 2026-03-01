import cv2
import mediapipe as mp
import numpy as np
import os
import time

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "gesture_data"
NUM_SAMPLES = 200

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)

print("💡 Press 'q' at any time to quit.")

while True:
    # Ask user for gesture label
    gesture_name = input("\nEnter the label for the gesture you want to capture (or 'exit' to quit): ").strip()
    if gesture_name.lower() == "exit":
        break

    # Create folder for this gesture
    os.makedirs(os.path.join(DATASET_PATH, gesture_name), exist_ok=True)
    sample_count = 0

    print(f"Get ready to perform gesture '{gesture_name}'...")
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    while sample_count < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Normalize landmarks relative to wrist (landmark 0)
            base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            landmarks = np.array(landmarks)

            # Save sample
            file_path = os.path.join(DATASET_PATH, gesture_name, f"{sample_count}.npy")
            np.save(file_path, landmarks)
            sample_count += 1

        # Display feedback
        cv2.putText(frame, f"{gesture_name}: {sample_count}/{NUM_SAMPLES}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Gesture Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"✅ Captured {sample_count} samples for '{gesture_name}'")

cap.release()
cv2.destroyAllWindows()
print("🎉 All done!")
