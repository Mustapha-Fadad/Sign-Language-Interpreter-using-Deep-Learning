import cv2
import mediapipe as mp
import pickle
import pyttsx3
import threading
import os
import sys
import numpy as np
from collections import deque, Counter

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

engine = pyttsx3.init()
engine.setProperty('rate', 150)
is_speaking = False

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

def speak_text(text):
    global is_speaking
    if not is_speaking:
        is_speaking = True
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
        is_speaking = False

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print("Erreur: Impossible de localiser le modele. Avez-vous lance src/train_model.py ?")
    sys.exit(1)

print("Chargement de l'intelligence artificielle...")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# AMELIORATION 6 : Lissage / Smoothing Buffer
# Memorise les 12 dernieres frames pour appliquer un "Vote Majoritaire"
# Cela elimine physiquement les petits "clignotements" (flickers) d'erreurs d'une seule frame
SMOOTHING_FRAMES = 12
prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)
previous_spoken = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = []
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
            
            if len(features) == 63:
                # Normalisation Spatiale
                features = normalize_landmarks(features)
                
                # Prediction Brute Locale
                pred_encoded = model.predict([features])[0]
                raw_pred = le.inverse_transform([pred_encoded])[0]
                
                # Confidence Score
                probs = model.predict_proba([features])[0]
                confidence = max(probs) * 100

                # Nous ajoutons uniquement les predictions viables au buffer temporel (> 45% confiance)
                if confidence > 45.0:
                    prediction_buffer.append(raw_pred)

                # AMELIORATION : Execution du lissage par Vote
                if len(prediction_buffer) > 0:
                    # Trouve le mot le plus frequent dans les dernieres frames
                    smooth_prediction = Counter(prediction_buffer).most_common(1)[0][0]
                    
                    color = (0, 255, 0) if confidence > 75 else (0, 165, 255)
                    text = f"{smooth_prediction} (Conf: {confidence:.0f}%)"
                    
                    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                    
                    # Declencheur vocal : uniquement si le buffer ENTIER s'accorde (certitude temporelle totale)
                    if prediction_buffer.count(smooth_prediction) >= (SMOOTHING_FRAMES - 2) and smooth_prediction != previous_spoken:
                        threading.Thread(target=speak_text, args=(smooth_prediction,), daemon=True).start()
                        previous_spoken = smooth_prediction
    else:
        # Re-set du filtre temporel si la main disparait
        prediction_buffer.clear()
        previous_spoken = None
        cv2.putText(frame, "Recherche de Main...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Inference Lissage Anti-Erreur (Appuyez sur Q pour quitter)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
