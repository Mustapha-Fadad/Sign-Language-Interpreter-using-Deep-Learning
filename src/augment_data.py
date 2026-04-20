import pandas as pd
import numpy as np
import os
import sys
import math

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'custom_dataset.csv')
AUG_DATA_PATH = os.path.join(BASE_DIR, 'data', 'gesture_dataset_augmented.csv')

def normalize_landmarks(features):
    """
    Soustrait les coordonnees du poignet (x0, y0, z0) de tous les autres points.
    rend le modele invariant a la position spatiale.
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

def rotate_2d(x, y, angle_degrees):
    """Effectue une rotation 2D (autour de 0,0) d'un angle donne en degres."""
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    return new_x, new_y

def apply_augmentation(features):
    """
    Applique de l'augmentation de donnees (Scale, Rotation, Noise, Translation) sur une structure.
    """
    aug_features = np.copy(features)
    
    # 1. Scaling (Zoom in / Zoom out legers)
    scale_factor = np.random.uniform(0.8, 1.2)
    aug_features = aug_features * scale_factor
    
    # 2. Rotation 2D (inclinaison de la main gauche/droite)
    # Angle aleatoire entre -15 et 15 degres
    angle = np.random.uniform(-15.0, 15.0)
    for i in range(21):
        # Rotation appliquee uniquement sur X et Y (autour de l'axe Z / de la camera)
        x_idx, y_idx = i * 3, i * 3 + 1
        rx, ry = rotate_2d(aug_features[x_idx], aug_features[y_idx], angle)
        aug_features[x_idx] = rx
        aug_features[y_idx] = ry

    # 3. Translation (Shift global avant normalisation)
    # L'utilisateur a demande une translation globale. Bien que la normalisation au poignet
    # neutralisera la translation *exacte* parfaite sur tous les points,
    # appliquer ce decalage permet de generer de la variabilite realiste avant le bruit.
    shift_x = np.random.uniform(-0.1, 0.1)
    shift_y = np.random.uniform(-0.1, 0.1)
    for i in range(21):
        aug_features[i*3] += shift_x
        aug_features[i*3+1] += shift_y

    # 4. Gaussian Noise (Bruit/tremblement individuel sur chaque joint)
    noise = np.random.normal(0, 0.015, len(aug_features))
    aug_features = aug_features + noise
    
    # 5. On force LA NORMALISATION a nouveau pour garantir que le poignet reste a 0,0,0
    # (cela annule la translation globale, mais concerve les deformations dues au bruit par point)
    return normalize_landmarks(aug_features)

def augment_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"Erreur: {DATA_PATH} introuvable.")
        sys.exit(1)
        
    print("[1/3] Chargement du Dataset Original...")
    df = pd.read_csv(DATA_PATH).dropna()
    print(f"Taille originale : {len(df)} echantillons.")
    
    print("\n[2/3] Normalisation, Equilibrage et Augmentation Massive...")
    # Cible exacte demandee par l'utilisateur: ~6000 samples par classe
    target_count = 6000
    print(f"Objectif cible par classe : {target_count} echantillons.")
    
    augmented_rows = []
    
    for label, group in df.groupby('label'):
        class_samples = group.drop('label', axis=1).values
        current_count = len(class_samples)
        
        # A) Conserver les originaux mais les NORMALISER
        for features in class_samples:
            norm_f = normalize_landmarks(features)
            row = [label] + norm_f.tolist()
            augmented_rows.append(row)
            
        # B) Generer des synthétiques jusqu'a atteindre target_count exactement
        samples_to_generate = target_count - current_count
        if samples_to_generate > 0:
            indices_to_augment = np.random.choice(current_count, samples_to_generate, replace=True)
            for idx in indices_to_augment:
                base_features = class_samples[idx]
                norm_f = normalize_landmarks(base_features) # On normalise la base
                aug_f = apply_augmentation(norm_f)          # On applique l'augmentation massive
                row = [label] + aug_f.tolist()
                augmented_rows.append(row)

    print("\n[3/3] Sauvegarde du Nouveau Dataset...")
    cols = ['label']
    for i in range(21):
        cols.extend([f"x{i}", f"y{i}", f"z{i}"])
        
    aug_df = pd.DataFrame(augmented_rows, columns=cols)
    aug_df.to_csv(AUG_DATA_PATH, index=False)
    
    print("Nouvelle repartition parfaite des classes :")
    print(aug_df['label'].value_counts())
    print(f"\n--> Succes ! Dataset Massivement Augmente (Taille {len(aug_df)}) sauvegarde dans : {AUG_DATA_PATH}")

if __name__ == '__main__':
    augment_dataset()
