# Custom Keypoint Sign Language Recognition System

A hand gesture recognition project using MediaPipe to extract 21 hand
landmarks and a Random Forest classifier to recognize custom gestures.

## Directory Structure

- `data/`: Recorded gesture keypoint CSV files.
- `models/`: Trained `rf_model.pkl` and `label_encoder.pkl`.
- `src/`: Data collection, training, and live prediction scripts.
- `archive/`: Deprecated scripts retained for reference.

## Pipeline

1. **Collect keypoints:** Record hand landmark coordinates from a webcam
   for each gesture.
2. **Train the model:** Train a Random Forest classifier on the collected data.
3. **Run live predictions:** Recognize gestures from the webcam using
   the trained model.

## Installation

pip install -r requirements.txt

## Usage

### 1. Collect custom gestures

python src/collect_keypoints.py

### 2. Train the model

python src/train_model.py

### 3. Run live webcam prediction

python src/final.py

## Acknowledgments

Based on the original project:
https://github.com/ismailzoukenissas/Sign-Language-Interpreter-using-Deep-Learning

## Scope

This system recognizes the gestures represented in its training data.
It does not translate complete sign language conversations.