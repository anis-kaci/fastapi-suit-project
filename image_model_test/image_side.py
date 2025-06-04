import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Charger ton modèle personnalisé
model = load_model("best_model_so_far.keras")

# Paramètres
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
input_size = (96, 96)  # adapte selon ton modèle

# Détecteur de visages Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_face(face):
    face_resized = cv2.resize(face, input_size)
    face_normalized = face_resized / 255.0
    return np.expand_dims(face_normalized, axis=0)  # shape (1, h, w, 3)

def extract_emotions_from_video(video_path, output_csv=None):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames // fps

    results = []

    for sec in range(duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convertir en niveaux de gris pour la détection de visages
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print(f"Aucun visage détecté à {sec}s")
            continue

        for (x, y, w, h) in faces:
            face_rgb = rgb[y:y+h, x:x+w]

            try:
                face_input = preprocess_face(face_rgb)
                predictions = model.predict(face_input, verbose=0)[0]
                emotion_scores = dict(zip(emotion_labels, predictions * 100))  # % des émotions

                print(f"Frame {sec}s :")
                for emo, score in emotion_scores.items():
                    print(f"  {emo}: {score:.2f}%")

                results.append((sec, emotion_scores))
                break  # un seul visage par frame

            except Exception as e:
                print(f"Erreur de prédiction à {sec}s : {str(e)}")
                continue

    cap.release()

    if output_csv and results:
        import csv
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['second'] + emotion_labels
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for sec, emo_dict in results:
                row = {'second': sec}
                row.update(emo_dict)
                writer.writerow(row)

    return results

# 🔁 Exemple d'appel
video_path = "../data/test_emotion.mov"  # Remplace par le nom de ta vidéo
extract_emotions_from_video(video_path, output_csv="emotions_output.csv")