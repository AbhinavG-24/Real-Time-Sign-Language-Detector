import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json

DATASET_PATH = "gesture_data"
MODEL_PATH = "gesture_model.h5"
LABEL_PATH = "label_map.json"

# -----------------------------
# LOAD DATA
# -----------------------------
X, y = [], []
gestures = sorted(os.listdir(DATASET_PATH))
gesture_map = {gesture: idx for idx, gesture in enumerate(gestures)}

# Print dataset info first
print("Detected gestures:", gestures)

total_samples = 0
for gesture in gestures:
    files = os.listdir(os.path.join(DATASET_PATH, gesture))
    print(f"{gesture} has {len(files)} samples")
    total_samples += len(files)

if total_samples == 0:
    print("❌ No data found. Please capture gestures first!")
    exit()

# Load data
for gesture in gestures:
    folder_path = os.path.join(DATASET_PATH, gesture)
    for file in os.listdir(folder_path):
        X.append(np.load(os.path.join(folder_path, file)))
        y.append(gesture_map[gesture])

X = np.array(X)
y = np.array(y)
print(f"✅ Total samples loaded: {len(X)}")

# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples")

# -----------------------------
# BUILD MODEL
# -----------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1],)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(gestures), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16
)

# -----------------------------
# SAVE MODEL & LABELS
# -----------------------------
model.save(MODEL_PATH)
with open(LABEL_PATH, "w") as f:
    json.dump(gesture_map, f)

print("✅ Model saved as", MODEL_PATH)
print("✅ Label mapping saved as", LABEL_PATH)
