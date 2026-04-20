import cv2
import mediapipe as mp
import os
import time
import csv

# Path config
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'custom_dataset.csv')

# MediaPipe config
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# AMELIORATION 1 : Collecte massive
FRAMES_TO_CAPTURE = 500

def initialize_csv():
    # Only write header if file doesn't exist
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print(f"Creation du fichier dataset : {DATA_PATH}")
        header = ["label"]
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        
        with open(DATA_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def collect_data():
    initialize_csv()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam.")
        return

    while True:
        # Prompt user via console
        label = input("\nEntrez le label du geste (ex: 'bonjour') ou 'q' pour quitter : ").strip()
        if label.lower() == 'q':
            break
            
        print(f"\\nPréparez-vous à faire le geste '{label}' !")
        for i in range(3, 0, -1):
            print(f"Démarrage dans {i}...")
            time.sleep(1)
            
        print(f"ENREGISTREMENT EN COURS ! Maintenez le geste. Objectif: {FRAMES_TO_CAPTURE} captures valides.")
        
        frames_captured = 0
        
        while frames_captured < FRAMES_TO_CAPTURE:
            ret, frame = cap.read()
            if not ret:
                break

            # Miroir
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand landmarks
            results = hands.process(rgb_frame)
            
            # AMELIORATION 6 : Verification de qualite 
            # On n'enregistre que si la main est completement trouvee
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extraction des 21 points
                    row = [label]
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                        
                    # Verification securite : 63 coordonnees + 1 label
                    if len(row) == 64:
                        with open(DATA_PATH, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(row)
                        
                        frames_captured += 1
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        # Afficher retour visuel de qualite OK
                        cv2.putText(frame, "Qualite Capture: OK", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                # Main masquee ou introuvable -> frame ignoree
                cv2.putText(frame, "Main introuvable - Frame ignoree", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # UI overlays
            cv2.putText(frame, f"Geste : {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Progression : {frames_captured}/{FRAMES_TO_CAPTURE}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Collecte Automatique 500 Samples', frame)
            
            # Echappement manuel
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Collecte interrompue.")
                break
                
        print(f"\nSUPER ! Collecte terminee pour '{label}'. {frames_captured} echantillons propres ont ete sauvegardes.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
