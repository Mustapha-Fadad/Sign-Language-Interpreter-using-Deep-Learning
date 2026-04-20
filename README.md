# Custom Keypoint Sign Language Recognition System

A clean, scratch-built Machine Learning project for Hand Gesture/Sign Language detection mapping pure 21-hand landmarks (via MediaPipe) through a standard random forest classifier.

## Directory Structure
- `data/`         : Location where your recorded keypoint CSVs will be saved.
- `models/`       : Location where your trained `rf_model.pkl` and `label_encoder.pkl` reside.
- `src/`          : The core logic of the application.
- `archive/`      : The location for your old legacy scripts that have been deprecated.

## Pipeline Flow

1. **Collect Keypoints**: Extract raw feature coordinates from your webcam per gesture.
2. **Train Model**: Fit a classical random forest classifier directly to your customized CSV.
3. **Live Test**: Deploy a live OpenCV window capturing identical keypoint extraction loops and live-predicting against the trained `rf_model.pkl`.

## How to execute

### Install Dependencies
`pip install -r requirements.txt`

### 1) Gather custom gestures
`python src/collect_keypoints.py`

### 2) Train model
`python src/train_model.py`

### 3) Live Test webcam prediction
`python src/final.py`
