from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import sys

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

print("Loading Machine Learning Model and Encoder...")
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print("Error: Model or encoder not found.")
    print("You must train the model first.")
    sys.exit(1)

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

def normalize_landmarks(features):
    """
    Soustrait les coordonnees du poignet (x0, y0, z0) de tous les autres points.
    rend le modele invariant a la position spatiale (translation).
    Idempotent : ne fera rien si les coordonnees du poignet sont deja (0,0,0).
    """
    features = np.array(features, dtype=float)
    if len(features) != 63:
        return features
        
    norm_features = np.zeros(63)
    x_base, y_base, z_base = features[0], features[1], features[2]
    
    for i in range(21):
        norm_features[i * 3] = features[i * 3] - x_base
        norm_features[i * 3 + 1] = features[i * 3 + 1] - y_base
        norm_features[i * 3 + 2] = features[i * 3 + 2] - z_base
        
    return norm_features

# MediaPipe Initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Global variables for state management
camera = None
is_camera_running = False

latest_prediction = {
    "label": "Waiting...",
    "confidence": 0.0
}

def generate_frames():
    global camera, is_camera_running, latest_prediction
    
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        is_camera_running = True

    while is_camera_running:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process via MediaPipe
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract 63 features
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
                # Predict
                if len(features) == 63:
                    features = normalize_landmarks(features)
                    prediction_encoded = model.predict([features])[0]
                    prediction = le.inverse_transform([prediction_encoded])[0]
                    
                    probabilities = model.predict_proba([features])[0]
                    confidence = max(probabilities) * 100
                    
                    # Update global prediction strictly for the frontend
                    latest_prediction['label'] = prediction
                    latest_prediction['confidence'] = round(confidence, 2)
        else:
            latest_prediction['label'] = "No Hand Detected"
            latest_prediction['confidence'] = 0.0

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame directly to browser over HTTP multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route connecting the frontend image tag to the generator."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict_feed')
def predict_feed():
    """Returns the latest prediction context in JSON format for realtime AJAX updates."""
    return jsonify(latest_prediction)

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Route to shut off the camera hardware gracefully."""
    global camera, is_camera_running, latest_prediction
    is_camera_running = False
    if camera is not None:
        camera.release()
        camera = None
    latest_prediction = {
        "label": "Camera Offline",
        "confidence": 0.0
    }
    return jsonify({"status": "stopped"})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
