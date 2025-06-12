# main.py

import whisper
import tempfile
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Depends, status, Response, Cookie, Header
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Dict
from typing import Optional, List
from sqlalchemy.orm import Session as DBSession
from passlib.context import CryptContext
from db.database import SessionLocal, engine, Base
from db.models import User, UserCredentials
from datetime import datetime
from db.models import Session, User # modèles pour User et Session
import uuid

import json # Import json for WebSocket communication
import logging # Import logging

# Imports for Sentiment Analysis
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydantic Models for API Responses ---
class TranscriptionResult(BaseModel):
    text: str

class FaceEmotion(BaseModel):
    bounding_box: dict
    emotions: dict
    sentiments: Optional[dict] = {}

class FacialAnalysisResult(BaseModel):
    facial_emotions: list[FaceEmotion]

class FrameEmotionDetail(BaseModel):
    timestamp_seconds: int
    detected_faces: list[FaceEmotion]

class VideoAnalysisResult(BaseModel):
    video_duration_seconds: int
    frames_analyzed: int
    emotions_timeline: list[FrameEmotionDetail]

class SentimentAnalysisResult(BaseModel):
    overall_sentiment: str
    confidence_score: float
    raw_scores: dict

# NEW Pydantic model for Text Emotion Analysis
class TextEmotionAnalysisResult(BaseModel):
    overall_emotion: str
    confidence_score: float
    raw_scores: dict

class InterviewAnalysisResult(BaseModel):
    transcription: TranscriptionResult
    video_emotions: VideoAnalysisResult
    overall_text_sentiment: SentimentAnalysisResult

# Pydantic model pour les données reçues
class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    birth_date: str
    education_level: str
    target_position: str
    email: EmailStr
    password: str


# DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Modified get_device function to always prefer GPU if available for Hugging Face models
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("cpu") # Forcing CPU for MPS as per original comment
    else:
        return torch.device("cpu")
    

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multimodal Chatbot API",
    description="API for audio transcription, facial emotion detection from images/videos, and text sentiment analysis.",
    version="1.0.0",
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_current_user(token: str = Depends(oauth2_scheme), db: DBSession = Depends(get_db)):
    # Ici tu vérifies ton token et récupères l'utilisateur
    # Exemple fictif:
    user = db.query(User).filter(User.token == token).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Non authentifié")
    return user

origins = [
    "*" # Allows all origins for local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
#--- pour la page d'accueil

@app.get("/home")
def home(authorization: str = Header(None), db: DBSession = Depends(get_db)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Non autorisé")

    # Ex: Authorization: Bearer <token>
    token = authorization.split(" ")[1] if " " in authorization else authorization

    session = db.query(Session).filter(Session.token == token).first()
    if not session:
        raise HTTPException(status_code=401, detail="Session invalide")

    user = session.user
    return {"user_name": user.first_name}

# ---- register a new user ----
@app.post("/register")
def register_user(data: RegisterRequest, db: DBSession = Depends(get_db)):
    # Vérifie si l'email existe déjà
    existing = db.query(UserCredentials).filter_by(email=data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Crée l'utilisateur
    user = User(
        first_name=data.first_name,
        last_name=data.last_name,
        birth_date=data.birth_date,
        education_level=data.education_level,
        target_position=data.target_position
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Crée les identifiants
    hashed_pw = pwd_context.hash(data.password)
    credentials = UserCredentials(
        user_id=user.id,
        email=data.email,
        hashed_password=hashed_pw
    )
    db.add(credentials)
    db.commit()

    return {"message": "User created successfully"}

# --- connexion ----
class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/login")
def login(data: LoginRequest, db: DBSession = Depends(get_db)):
    user_cred = db.query(UserCredentials).filter(UserCredentials.email == data.email).first()
    if not user_cred or not pwd_context.verify(data.password, user_cred.hashed_password):
        raise HTTPException(status_code=400, detail="Email ou mot de passe incorrect")

    # Créer une session avec token unique
    new_session = Session(
        user_id=user_cred.user_id, # ou user_cred.id selon ta structure
        token=str(uuid.uuid4()), 
        created_at=datetime.utcnow()
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)

    return {"message": "Connexion réussie", "token": new_session.token}

# --- Global Model Loading (for performance) ---

# Whisper Model (loaded on demand for 'base' model, as it's efficient enough)
WHISPER_MODEL = "base"

# Facial Analysis Model (your best_model_so_far.keras)
EMOTION_FACE_MODEL_PATH = "models/best_model_emotion.keras" # Make sure this file is in your project root or provide the full path
SENTIMENT_IMAGE_MODEL_PATH = "models/best_model_sentiment.keras" # This seems to be for image sentiment, not text
EMOTION_FACE_MODEL = None
SENTIMENT_IMAGE_MODEL = None # This will remain for facial sentiment
FACE_DETECTOR = None # Initialize face detector

try:
    EMOTION_FACE_MODEL = load_model(EMOTION_FACE_MODEL_PATH)
    logger.info(f"Loaded facial emotion model: {EMOTION_FACE_MODEL_PATH}")

    # Load Haar Cascade Classifier for face detection
    FACE_DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if FACE_DETECTOR.empty():
        logger.warning("WARNING: Haar Cascade for face detection not loaded. Facial analysis might fail.")
except Exception as e:
    logger.error(f"Error loading facial emotion model or face detector: {e}. Ensure '{EMOTION_FACE_MODEL_PATH}' exists and TensorFlow/OpenCV are installed.")
    EMOTION_FACE_MODEL = None
    FACE_DETECTOR = None
    
try:
    SENTIMENT_IMAGE_MODEL = load_model(SENTIMENT_IMAGE_MODEL_PATH)
    logger.info(f"Loaded facial sentiment model: {SENTIMENT_IMAGE_MODEL_PATH}") # Corrected print message
    # Face detector is already attempted to be loaded above, no need to repeat
except Exception as e:
    logger.error(f"Error loading facial sentiment model: {e}. Ensure '{SENTIMENT_IMAGE_MODEL_PATH}' exists and TensorFlow is installed.")
    SENTIMENT_IMAGE_MODEL = None


# Define facial emotion labels and input size globally, based on your model
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
MODEL_INPUT_SIZE = (96, 96) # Height, Width

SENTIMENT_IMAGE_LABELS = ['Negative', 'Neutral', 'Positive']

# --- Text Sentiment Analysis Model (Fine-tuned GPT-2 from local folder) ---
GPT2_SENTIMENT_MODEL_PATH = "models/best_model_sentiment" 

text_sentiment_tokenizer = None
text_sentiment_model = None
TEXT_SENTIMENT_LABELS = [] # Will be populated from the model's config

# --- Text Emotion Analysis Model (Fine-tuned GPT-2 from local folder) ---
GPT2_EMOTION_MODEL_PATH = "models/best_model_emotion" # This is the path for your text emotion model

text_emotion_tokenizer = None
text_emotion_model = None
# YOU MUST DEFINE THESE LABELS FOR YOUR EMOTION MODEL!
# Based on your config.json having LABEL_0 to LABEL_6, you need to know what each of these means.
# EXAMPLE:
# For example, if your model was trained on these 7 emotions in this order:
TEXT_EMOTION_LABELS = [
    "anger",    # Corresponds to LABEL_0
    "disgust",  # Corresponds to LABEL_1
    "fear",     # Corresponds to LABEL_2
    "joy",      # Corresponds to LABEL_3
    "neutral",  # Corresponds to LABEL_4
    "sadness",  # Corresponds to LABEL_5
    "surprise"  # Corresponds to LABEL_6
]
# Adjust the above list to match the exact order and names of your emotions!


# --- Model and Tokenizer Initialization in startup ---
@app.on_event("startup")
async def load_models_on_startup():
    global text_sentiment_tokenizer, text_sentiment_model, TEXT_SENTIMENT_LABELS
    global text_emotion_tokenizer, text_emotion_model, TEXT_EMOTION_LABELS # No global for TEXT_EMOTION_LABELS if manually defined

    device = get_device()
    logger.info(f"Loading Hugging Face models on device: {device}")

    # Load Text Sentiment Model
    try:
        logger.info(f"Loading text sentiment model from '{GPT2_SENTIMENT_MODEL_PATH}'...")
        text_sentiment_tokenizer = AutoTokenizer.from_pretrained(GPT2_SENTIMENT_MODEL_PATH)
        text_sentiment_model = AutoModelForSequenceClassification.from_pretrained(GPT2_SENTIMENT_MODEL_PATH)
        text_sentiment_model.to(device)
        text_sentiment_model.eval() # Set model to evaluation mode

        if hasattr(text_sentiment_model.config, 'id2label') and text_sentiment_model.config.id2label:
            # Sort labels by their ID to ensure consistent order
            label_map = sorted(text_sentiment_model.config.id2label.items(), key=lambda x: int(x[0]))
            TEXT_SENTIMENT_LABELS = [label for _, label in label_map]
            logger.info(f"Text sentiment model labels from config: {TEXT_SENTIMENT_LABELS}")
        else:
            TEXT_SENTIMENT_LABELS = ["negative", "neutral", "positive"] 
            logger.warning(f"WARNING: Could not auto-detect text sentiment labels from model config. Using default: {TEXT_SENTIMENT_LABELS}")
        
        logger.info(f"Loaded fine-tuned GPT-2 sentiment model from: {GPT2_SENTIMENT_MODEL_PATH} with labels: {TEXT_SENTIMENT_LABELS}")

    except Exception as e:
        logger.error(f"Error loading fine-tuned GPT-2 sentiment model from '{GPT2_SENTIMENT_MODEL_PATH}': {e}.")
        # Optionally re-raise or set a flag to prevent endpoint usage
        text_sentiment_tokenizer = None
        text_sentiment_model = None
        # Do not raise here to allow other models to load if this one fails

    # Load Text Emotion Model
    try:
        logger.info(f"Loading text emotion model from '{GPT2_EMOTION_MODEL_PATH}'...")
        text_emotion_tokenizer = AutoTokenizer.from_pretrained(GPT2_EMOTION_MODEL_PATH)
        text_emotion_model = AutoModelForSequenceClassification.from_pretrained(GPT2_EMOTION_MODEL_PATH)
        text_emotion_model.to(device)
        text_emotion_model.eval() # Set model to evaluation mode

        # If your emotion model's config.json has "id2label" with actual names, use it.
        # Otherwise, stick to the manually defined TEXT_EMOTION_LABELS.
        if hasattr(text_emotion_model.config, 'id2label') and text_emotion_model.config.id2label:
            # Check if id2label contains human-readable names or just LABEL_X
            first_label_value = next(iter(text_emotion_model.config.id2label.values()))
            if not first_label_value.startswith("LABEL_"): # If they are like "anger", "joy"
                label_map = sorted(text_emotion_model.config.id2label.items(), key=lambda x: int(x[0]))
                TEXT_EMOTION_LABELS = [label for _, label in label_map]
                logger.info(f"Text emotion model labels from config: {TEXT_EMOTION_LABELS}")
            else: # Still LABEL_X, so we use our hardcoded list
                logger.warning(f"Text emotion model config.id2label found generic labels ({first_label_value}). Using manually defined TEXT_EMOTION_LABELS: {TEXT_EMOTION_LABELS}")
        else:
            logger.warning(f"Text emotion model config.id2label not found. Using manually defined TEXT_EMOTION_LABELS: {TEXT_EMOTION_LABELS}")
        
        logger.info(f"Loaded fine-tuned GPT-2 emotion model from: {GPT2_EMOTION_MODEL_PATH} with labels: {TEXT_EMOTION_LABELS}")

    except Exception as e:
        logger.error(f"Error loading fine-tuned GPT-2 emotion model from '{GPT2_EMOTION_MODEL_PATH}': {e}.")
        text_emotion_tokenizer = None
        text_emotion_model = None


# --- Helper Functions ---

async def transcribe_audio_with_whisper(audio_file_path: str) -> str:
    """
    Transcribes an audio file (or audio track from a video) using the local Whisper model.
    """
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(audio_file_path) # Whisper can handle video files to extract audio
        return result["text"]
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Whisper transcription failed: {e}. Ensure FFmpeg is installed and in PATH.")


def preprocess_face_for_model(face_rgb_image):
    """
    Resizes and normalizes a cropped RGB face image for the CNN model.
    Expected input: NumPy array (H, W, 3) representing a face in RGB.
    Expected output: NumPy array (1, target_H, target_W, 3) ready for model.predict().
    """
    face_resized = cv2.resize(face_rgb_image, MODEL_INPUT_SIZE)
    face_normalized = face_resized / 255.0
    return np.expand_dims(face_normalized, axis=0) # Add batch dimension


async def analyze_facial_emotions(image_bytes: bytes) -> list[dict]:
    """
    Analyzes facial emotions from a single image's bytes.
    Returns a list of dicts with bounding box and all emotion probabilities.
    """
    if EMOTION_FACE_MODEL is None or FACE_DETECTOR is None:
        logger.warning("Facial emotion model or detector not loaded. Cannot perform analysis.")
        return []

    try:
        np_image = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Could not decode image file.")
            return []

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = FACE_DETECTOR.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        detected_emotions = []
        if len(faces) == 0:
            return []

        for (x, y, w, h) in faces:
            face_rgb_roi = rgb_img[y:y+h, x:x+w]

            if face_rgb_roi.size == 0:
                continue

            try:
                face_input = preprocess_face_for_model(face_rgb_roi)
                predictions = EMOTION_FACE_MODEL.predict(face_input, verbose=0)[0]

                emotion_probabilities = {
                    emotion: float(prob) for emotion, prob in zip(EMOTION_LABELS, predictions)
                }

                detected_emotions.append({
                    "bounding_box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    },
                    "emotions": emotion_probabilities
                })

            except Exception as e:
                logger.error(f"Error predicting emotion for a face: {e}")
                continue

        return detected_emotions

    except Exception as e:
        logger.error(f"Facial analysis internal error: {e}")
        return []


async def analyze_facial_sentiments(image_bytes: bytes) -> list[dict]:
    """
    Analyzes facial sentiments from an image.
    Returns a list of dicts with bounding box and sentiment probabilities.
    """
    if SENTIMENT_IMAGE_MODEL is None or FACE_DETECTOR is None:
        logger.warning("Sentiment model or face detector not loaded. Cannot perform analysis.")
        return []

    try:
        np_image = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("Could not decode image file.")
            return []

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = FACE_DETECTOR.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        detected_sentiments = []
        if len(faces) == 0:
            return []

        for (x, y, w, h) in faces:
            face_rgb_roi = rgb_img[y:y+h, x:x+w]

            if face_rgb_roi.size == 0:
                continue

            try:
                # Resize and normalize face
                face_resized = cv2.resize(face_rgb_roi, MODEL_INPUT_SIZE)
                face_normalized = face_resized.astype("float32") / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)  # Add batch dimension

                predictions = SENTIMENT_IMAGE_MODEL.predict(face_input, verbose=0)[0]

                sentiment_probabilities = {
                    sentiment: float(prob)
                    for sentiment, prob in zip(SENTIMENT_IMAGE_LABELS, predictions)
                }

                detected_sentiments.append({
                    "bounding_box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    },
                    "sentiments": sentiment_probabilities
                })

            except Exception as e:
                logger.error(f"Error predicting sentiment for a face: {e}")
                continue

        return detected_sentiments

    except Exception as e:
        logger.error(f"Facial sentiment analysis internal error: {e}")
        return []


async def process_video_for_emotions(video_file_path: str) -> VideoAnalysisResult:
    """
    Processes a video file frame by frame (sampling one per second) to analyze facial emotions and sentiments.
    Returns all emotion and sentiment probabilities for each face detected in each frame.
    """
    if EMOTION_FACE_MODEL is None or SENTIMENT_IMAGE_MODEL is None or FACE_DETECTOR is None:
        raise HTTPException(status_code=500, detail="Required models or detector not loaded.")

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = int(total_frames / fps) if fps > 0 else 0

    emotions_timeline = []
    frames_processed = 0

    for sec in range(video_duration_seconds + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()

        if not ret:
            logger.info(f"Skipping frame at {sec} seconds.")
            continue

        frames_processed += 1

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = FACE_DETECTOR.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        detected_faces_for_frame = []

        for (x, y, w, h) in faces:
            face_rgb_roi = rgb_img[y:y+h, x:x+w]
            if face_rgb_roi.size == 0:
                continue

            try:
                # Emotion analysis
                face_input_emotion = preprocess_face_for_model(face_rgb_roi)
                emotion_preds = EMOTION_FACE_MODEL.predict(face_input_emotion, verbose=0)[0]
                emotion_probs = {
                    EMOTION_LABELS[i]: float(pred) for i, pred in enumerate(emotion_preds)
                }

                # Sentiment analysis
                face_resized = cv2.resize(face_rgb_roi, MODEL_INPUT_SIZE)
                face_normalized = face_resized.astype("float32") / 255.0
                face_input_sentiment = np.expand_dims(face_normalized, axis=0)
                sentiment_preds = SENTIMENT_IMAGE_MODEL.predict(face_input_sentiment, verbose=0)[0]
                sentiment_probs = {
                    SENTIMENT_IMAGE_LABELS[i]: float(pred) for i, pred in enumerate(sentiment_preds)
                }

                detected_faces_for_frame.append(FaceEmotion(
                    bounding_box={"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    emotions=emotion_probs,
                    sentiments=sentiment_probs
                ))

            except Exception as e:
                logger.error(f"Prediction error in frame {sec}s: {e}")
                continue

        emotions_timeline.append(FrameEmotionDetail(
            timestamp_seconds=sec,
            detected_faces=detected_faces_for_frame
        ))

    cap.release()

    return VideoAnalysisResult(
        video_duration_seconds=video_duration_seconds,
        frames_analyzed=frames_processed,
        emotions_timeline=emotions_timeline
    )


async def analyze_text_sentiment_logic(text: str) -> SentimentAnalysisResult:
    """
    Analyzes sentiment of text using the locally loaded Hugging Face GPT-2 model.
    """
    if text_sentiment_model is None or text_sentiment_tokenizer is None or not TEXT_SENTIMENT_LABELS:
        raise HTTPException(status_code=503, detail="Text sentiment analysis model not loaded or labels not defined. Check server logs.")

    try:
        device = get_device()
        inputs = text_sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = text_sentiment_model(**inputs)
        
        # Softmax to get probabilities
        probabilities = torch.softmax(outputs.logits, dim=1)[0] # Get probabilities for the single batch item

        # Get the predicted label and its confidence
        predicted_index = torch.argmax(probabilities).item()
        predicted_label = TEXT_SENTIMENT_LABELS[predicted_index]
        confidence_score = probabilities[predicted_index].item()

        # Get raw scores for all labels
        raw_scores = {}
        for i, prob in enumerate(probabilities):
            if i < len(TEXT_SENTIMENT_LABELS):
                raw_scores[TEXT_SENTIMENT_LABELS[i]] = prob.item()
            else:
                raw_scores[f"LABEL_{i}"] = prob.item() # Fallback, should not happen if labels are correctly populated

        return SentimentAnalysisResult(
            overall_sentiment=predicted_label,
            confidence_score=confidence_score,
            raw_scores=raw_scores
        )

    except Exception as e:
        logger.error(f"Detailed text sentiment analysis error with local GPT-2 model: {e}")
        raise HTTPException(status_code=500, detail=f"Text sentiment analysis failed: {e}")

async def analyze_text_emotion_logic(text: str) -> TextEmotionAnalysisResult:
    """
    Analyzes emotions of text using the locally loaded Hugging Face GPT-2 emotion model.
    """
    if text_emotion_model is None or text_emotion_tokenizer is None or not TEXT_EMOTION_LABELS:
        raise HTTPException(status_code=503, detail="Text emotion analysis model not loaded or labels not defined. Check server logs.")

    try:
        device = get_device()
        inputs = text_emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = text_emotion_model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

        predicted_index = torch.argmax(probabilities).item()
        predicted_label = TEXT_EMOTION_LABELS[predicted_index] # Use the manually defined labels
        confidence_score = probabilities[predicted_index].item()

        raw_scores = {}
        for i, prob in enumerate(probabilities):
            if i < len(TEXT_EMOTION_LABELS):
                raw_scores[TEXT_EMOTION_LABELS[i]] = prob.item()
            else:
                raw_scores[f"LABEL_{i}"] = prob.item()

        return TextEmotionAnalysisResult(
            overall_emotion=predicted_label,
            confidence_score=confidence_score,
            raw_scores=raw_scores
        )

    except Exception as e:
        logger.error(f"Detailed text emotion analysis error with local GPT-2 model: {e}")
        raise HTTPException(status_code=500, detail=f"Text emotion analysis failed: {e}")


# --- API Endpoints ---

# --- Audio Transcription Endpoint ---
@app.post("/transcribe", response_model=TranscriptionResult, summary="Transcribe Audio File")
async def transcribe_audio(audio: UploadFile = File(..., description="Audio file to transcribe")):
    """
    Transcribes an uploaded audio file into text using the local Whisper model.
    """
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.filename.split('.')[-1]}") as temp_audio_file:
        content = await audio.read()
        temp_audio_file.write(content)
        temp_file_path = temp_audio_file.name

    try:
        transcription = await transcribe_audio_with_whisper(temp_file_path)
        return TranscriptionResult(text=transcription)
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# --- Facial Analysis (Image) Endpoint ---
@app.post("/analyze-face", response_model=FacialAnalysisResult, summary="Analyze Facial Emotions and Sentiments from Image")
async def analyze_face(image: UploadFile = File(..., description="Image file to analyze facial emotions and sentiments")):
    """
    Detects faces in an uploaded image and predicts their emotions and sentiments using two separate models.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are supported (e.g., JPEG, PNG).")

    image_bytes = await image.read()

    emotions = await analyze_facial_emotions(image_bytes)
    sentiments = await analyze_facial_sentiments(image_bytes)

    if not emotions and not sentiments:
        return FacialAnalysisResult(facial_emotions=[])

    # Combine emotions and sentiments based on bounding box match
    combined_results = []

    for emotion_face in emotions:
        matching_sentiment = next(
            (s for s in sentiments if s["bounding_box"] == emotion_face["bounding_box"]),
            None
        )
        combined_results.append({
            "bounding_box": emotion_face["bounding_box"],
            "emotions": emotion_face.get("emotions", {}),
            "sentiments": matching_sentiment.get("sentiments", {}) if matching_sentiment else {}
        })

    return FacialAnalysisResult(facial_emotions=combined_results)


# --- Facial Analysis (Video) Endpoint ---
@app.post("/analyze-video-emotions", response_model=VideoAnalysisResult, summary="Analyze Facial Emotions from Video")
async def analyze_video_emotions(video: UploadFile = File(..., description="Video file to analyze facial emotions. Note: Video processing can be time-consuming.")):
    """
    Processes an uploaded video file to detect faces and analyze their emotions frame by frame (one frame per second).
    Returns a timeline of detected emotions throughout the video.
    """
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only video files are supported (e.g., MP4, MOV, AVI).")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{video.filename.split('.')[-1]}") as temp_video_file:
        content = await video.read()
        temp_video_file.write(content)
        temp_file_path = temp_video_file.name

    try:
        analysis_result = await process_video_for_emotions(temp_file_path)
        return analysis_result
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# --- Sentiment Analysis (Text) Endpoint ---
@app.post("/analyze-text-sentiment", response_model=SentimentAnalysisResult, summary="Analyze Sentiment of Text")
async def analyze_text_sentiment(text: str = Form(..., description="Text string to analyze sentiment for")):
    """
    Analyzes the sentiment of provided text using a fine-tuned GPT-2 model.
    Returns the overall sentiment, its confidence score, and raw scores for all possible labels.
    """
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for sentiment analysis.")
    
    sentiment_result = await analyze_text_sentiment_logic(text)
    return sentiment_result

# --- NEW: Text Emotion Analysis Endpoint ---
@app.post("/analyze-text-emotion", response_model=TextEmotionAnalysisResult, summary="Analyze Emotion of Text")
async def analyze_text_emotion(text: str = Form(..., description="Text string to analyze emotion for")):
    """
    Analyzes the emotion of provided text using a fine-tuned GPT-2 emotion model.
    Returns the overall emotion, its confidence score, and raw scores for all possible labels.
    """
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for emotion analysis.")
    
    emotion_result = await analyze_text_emotion_logic(text)
    return emotion_result


# --- NEW: Analyze Interview Video Endpoint ---
@app.post("/analyze-interview-video", response_model=InterviewAnalysisResult, summary="Analyze a full interview video (Audio + Facial + Text Sentiment)")
async def analyze_interview_video(video: UploadFile = File(..., description="Interview video file for comprehensive analysis. This process can be very time-consuming.")):
    """
    Processes an uploaded interview video to perform:
    1. Audio transcription using Whisper.
    2. Facial emotion analysis frame-by-frame.
    3. Sentiment analysis of the transcribed text.
    Combines all results into a single comprehensive response.
    """
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only video files are supported (e.g., MP4, MOV, AVI).")

    # Save the uploaded video file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{video.filename.split('.')[-1]}") as temp_video_file:
        content = await video.read()
        temp_video_file.write(content)
        temp_file_path = temp_video_file.name

    try:
        # Step 1: Perform Audio Transcription
        # Whisper can extract audio directly from video files via FFmpeg
        transcription_text = await transcribe_audio_with_whisper(temp_file_path)
        transcription_result = TranscriptionResult(text=transcription_text)
        logger.info(f"Transcription complete: {transcription_text[:50]}...") # Log first 50 chars

        # Step 2: Perform Facial Analysis from Video
        video_emotions_result = await process_video_for_emotions(temp_file_path)
        logger.info(f"Facial analysis complete for {video_emotions_result.frames_analyzed} frames.")

        # Step 3: Analyze Sentiment of the Transcribed Text
        overall_text_sentiment_result = await analyze_text_sentiment_logic(transcription_text)
        logger.info(f"Text sentiment analysis complete: {overall_text_sentiment_result.overall_sentiment}")

        return InterviewAnalysisResult(
            transcription=transcription_result,
            video_emotions=video_emotions_result,
            overall_text_sentiment=overall_text_sentiment_result
        )
    except HTTPException as e:
        # Re-raise HTTPExceptions from helper functions directly
        raise e
    except Exception as e:
        # Catch any other unexpected errors during orchestration
        logger.error(f"Error during full interview video analysis for {video.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze interview video: {e}")
    finally:
        # Clean up the temporary video file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            logger.info(f"Cleaned up temporary video file: {temp_file_path}")

# --- NEW: WebSocket Endpoint for Real-time Webcam Analysis ---
@app.websocket("/ws/analyze-webcam")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established for real-time webcam analysis.")
    try:
        while True:
            # Receive binary data (image frame) from the client
            image_bytes = await websocket.receive_bytes()
            
            # Perform facial emotion analysis on the received frame
            detected_faces = await analyze_facial_emotions(image_bytes)

            # Prepare the response data. You can expand this to include other real-time metrics
            # like speech rate, gaze, etc., if you implement them.
            response_data = {
                "facial_emotions": [face.model_dump() for face in detected_faces] # Convert Pydantic models to dicts
            }
            
            # Send the analysis result back to the client as JSON
            await websocket.send_json(response_data)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected from client.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Optionally send an error message to the client before closing
        await websocket.send_json({"error": str(e), "message": "An error occurred during real-time analysis."})
    finally:
        logger.info("WebSocket connection closed.")


# --- Root Endpoint for API testing ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Multimodal Chatbot API! Visit /docs for API documentation."}