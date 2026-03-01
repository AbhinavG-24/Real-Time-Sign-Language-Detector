# 🤟 Sign Language & Gesture Recognition

A real-time hand gesture recognition system using **MediaPipe** and **TensorFlow/Keras**. The model detects hand landmarks via webcam and classifies gestures into letters (A–L) and numbers (1–7) in real time.

---

## 📁 Project Structure

```
├── hand_tracking_dataset.py   # Capture gesture samples using webcam
├── train_gesture_model.py     # Train the gesture classification model
├── real_time_detector.py      # Run real-time gesture detection
├── label_map.json             # Gesture-to-label mapping
├── gesture_data/              # Captured gesture samples (not pushed to GitHub)
└── gesture_model.h5           # Trained model (not pushed to GitHub)
```

---

## 🧠 How It Works

1. **Data Collection** — MediaPipe detects 21 hand landmarks from webcam feed. Landmarks are normalized relative to the wrist and saved as `.npy` files.
2. **Model Training** — A simple Dense Neural Network is trained on the collected landmarks to classify gestures.
3. **Real-Time Detection** — The trained model runs live on webcam input and displays the predicted gesture on screen.

---

## 🖐️ Supported Gestures

| Letters | Numbers |
|---------|---------|
| A, B, C, D, E, F, G, H, I, K, L | 1, 2, 3, 4, 5, 6, 7 |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/gesture-recognition.git
cd gesture-recognition

# Install dependencies
pip install opencv-python mediapipe tensorflow scikit-learn numpy
```

---

## 🚀 Usage

### Step 1: Collect Gesture Data
```bash
python hand_tracking_dataset.py
```
- Enter a gesture label when prompted (e.g. `A`, `1`, etc.)
- Perform the gesture in front of your webcam
- 200 samples will be captured per gesture
- Press `q` to quit anytime

### Step 2: Train the Model
```bash
python train_gesture_model.py
```
- Trains a neural network on your captured data
- Saves `gesture_model.h5` and `label_map.json`

### Step 3: Run Real-Time Detection
```bash
python real_time_detector.py
```
- Opens your webcam and detects gestures in real time
- Press `q` to quit

---

## 🏗️ Model Architecture

```
Input (63 features: 21 landmarks × 3 coordinates)
→ Dense(128, ReLU) → Dropout(0.2)
→ Dense(64, ReLU)  → Dropout(0.2)
→ Dense(num_classes, Softmax)
```

- **Optimizer:** Adam  
- **Loss:** Sparse Categorical Crossentropy  
- **Epochs:** 50 | **Batch Size:** 16

---

## 📋 Requirements

- Python 3.8+
- Webcam
- opencv-python
- mediapipe
- tensorflow
- scikit-learn
- numpy

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) by Google for hand landmark detection
- [TensorFlow/Keras](https://www.tensorflow.org/) for model training
