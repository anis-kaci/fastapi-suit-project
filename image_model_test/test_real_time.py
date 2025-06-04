import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Charger le modèle entraîné
model = tf.keras.models.load_model("best_model_so_far.keras")

# Définir les labels des émotions
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Charger le classificateur de visages d'OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)  # Utilise la webcam (0 pour la webcam principale)

while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en niveaux de gris
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extraire la région du visage
        roi_gray = frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (96, 96))  # Redimensionner à la taille d'entrée du modèle
        roi_gray = roi_gray / 255.0  # Normaliser les valeurs de pixel
        roi_gray = np.expand_dims(img_to_array(roi_gray), axis=0)  # Préparer pour la prédiction

        # Prédire l'expression faciale
        prediction = model.predict(roi_gray)
        label = class_labels[np.argmax(prediction)]  # Récupérer l'étiquette prédite

        # Dessiner un rectangle autour du visage et afficher le label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Afficher l'image avec les prédictions
    cv2.imshow("Facial Expression Recognition", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
