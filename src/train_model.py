import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# Par defaut on utilise le dataset augmente
AUG_DATA_PATH = os.path.join(BASE_DIR, 'data', 'gesture_dataset_augmented.csv')
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, 'data', 'custom_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rf_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')

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

def train():
    data_path = AUG_DATA_PATH
    if not os.path.exists(data_path):
        print(f"Info: {AUG_DATA_PATH} introuvable.")
        print("Bascule automatique vers le dataset original non-augmente...")
        data_path = ORIGINAL_DATA_PATH

    if not os.path.exists(data_path):
        print("Erreur: Aucun dataset (original ou augmente) introuvable. Veuillez collecter des donnees d'abord.")
        sys.exit(1)

    print(f"\n[1/6] Chargement de la base de donnees : {os.path.basename(data_path)}")
    df = pd.read_csv(data_path)
    df = df.dropna()
    
    print("\n--- Repartition Initiale ---")
    print(df['label'].value_counts())

    if len(df['label'].unique()) < 2:
        print("\nErreur: Impossible de classifier. Vous avez besoin d'au moins 2 gestes differents enregistres.")
        sys.exit(1)

    # Equilibre des donnees
    print("\n[2/6] Equilibrage strict des donnees pour eviter les biais...")
    min_count = df['label'].value_counts().min()
    print(f"Limitation synchronisee a {min_count} echantillons par geste.")
    
    df = df.groupby('label').sample(n=min_count, random_state=42).reset_index(drop=True)
    
    X_raw = df.drop('label', axis=1).values
    y = df['label']

    print("\n[3/6] Normalisation Spatiale des coordonnees relatives au poignet...")
    X = []
    for row in X_raw:
        X.append(normalize_landmarks(row))
    X = np.array(X)

    print("\n[4/6] Encodage algorithmique des textes...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("\n[5/6] Separation entrainement/test (80/20) et Entrainement Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n[6/6] Analyse et Validation du Modele...\n")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("="*50)
    print(f"EXACTITUDE (ACCURACY) : {accuracy * 100:.2f}%")
    print("="*50)
    
    print("\nRAPPORT DE CLASSIFICATION (Precision & Rappel par geste) :")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("MATRICE DE CONFUSION (Affichage des erreurs croisees) :")
    print("(Les lignes sont les 'vrais' gestes, les colonnes sont les 'predictions' du bot)")
    print(confusion_matrix(y_test, y_pred))

    print("\n---> Sauvegarde et Serialsation du modele...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
        
    print(f"Enregistrement reussi: '{MODEL_PATH}'")

if __name__ == "__main__":
    train()
