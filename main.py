# main.py

import whisper
import ffmpeg
import tempfile
import os
import cv2
from pathlib import Path
import numpy as np
import tensorflow as tf
from sqlalchemy import cast, Float
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Depends, status, Response, Cookie, Header, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, EmailStr, Field
from typing import Dict
from typing import Optional, Any, List
from sqlalchemy.orm import Session as DBSession
from passlib.context import CryptContext
from db.database import SessionLocal, engine, Base
from db.models import User, UserCredentials, FacialAnalysis, TextSentiment
from db.results import save_facial_analysis, save_transcription, save_text_sentiment, save_question
from datetime import datetime
from db.models import Session, User, Transcription, StressAnalysis
from db.models import QuestionsBank # modèles pour User et Session
from db.requests import get_sessions_by_user_id, get_user_stress_history
from db.requests import get_user_questions
import uuid
import json # Import json for WebSocket communication
import logging # Import logging
# Imports for Sentiment Analysis
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sqlalchemy import func, desc
from datetime import datetime, timedelta
import subprocess
import asyncio
from collections import Counter

import joblib
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimelineEntry(BaseModel):
    timestamp: int
    emotion: Optional[str]
    sentiment: Optional[str]

'''class AnalysisRequest(BaseModel):
    session_id: int
    dominant_emotion: Optional[str]
    emotion_timeline: List[TimelineEntry]
    dominant_sentiment: Optional[str]
    sentiment_timeline: Optional[List[TimelineEntry]]
    duration: float
    frames: int'''

class TimelineEntry(BaseModel):
    timestamp: float
    emotion: str
    sentiment: str

class AnalysisRequest(BaseModel):
    session_id: int
    dominant_emotion: str
    emotion_timeline: List[Dict[str, Any]]
    dominant_sentiment: str
    sentiment_timeline: List[Dict[str, Any]]
    duration: float
    frames: int

class QuestionCreate(BaseModel):
    question_text: str
    user_id: Optional[int] = None  # Facultatif

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

# NEW: Pydantic model for XGBoost Audio Emotion Result
class XGBoostAudioEmotionResult(BaseModel):
    dominant_emotion: Optional[str] = None
    confidence: Optional[float] = None
    all_scores: Dict[str, float] = {}
    message: str = ""

# UPDATED: Pydantic model for Realtime Combined Analysis (LLM audio removed)
class RealtimeCombinedAnalysis(BaseModel):
    facial_analysis: List[FaceEmotion] = []
    transcription: str = ""
    xgb_audio_emotion: XGBoostAudioEmotionResult = Field(default_factory=XGBoostAudioEmotionResult) # Added XGBoost result
    text_sentiment: SentimentAnalysisResult = Field(default_factory=lambda: SentimentAnalysisResult(
        overall_sentiment="N/A", confidence_score=0.0, raw_scores={}))

class InterviewAnalysisResult(BaseModel):
    transcription: TranscriptionResult
    video_emotions: VideoAnalysisResult
    overall_text_sentiment: SentimentAnalysisResult
    xgb_audio_emotion: XGBoostAudioEmotionResult # NEW FIELD FOR XGBOOST AUDIO EMOTION
    multimodal_sentiment: SentimentAnalysisResult # NEW FIELD FOR AGGREGATED SENTIMENT
    multimodal_emotion: XGBoostAudioEmotionResult # NEW FIELD FOR AGGREGATED EMOTION (reusing XGBoost model to hold label and scores)



class TextEmotionAnalysisResult(BaseModel):
    overall_emotion: str
    confidence_score: float
    raw_scores: Dict[str, float]
# Pydantic model pour les données reçues
class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    birth_date: str
    education_level: str
    target_position: str
    email: EmailStr
    password: str

class QuestionOut(BaseModel):
    question_id: int
    question_text: str
    user_id: Optional[int] = None

    class Config:
        orm_mode = True

# --- Schémas Pydantic ---

class StressAnalysisCreate(BaseModel):
    session_id: int
    stress_facial: Optional[float] = None
    stress_textuel: Optional[float] = None
    stress_global: Optional[float] = None

class StressAnalysisRead(BaseModel):
    id: int
    user_id: int
    session_id: int
    stress_facial: Optional[float]
    stress_textuel: Optional[float]
    stress_global: Optional[float]
    created_at: datetime

    class Config:
        orm_mode = True
        
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
    

def bytes_to_cv2_image(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR numpy array
    return img_np

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multimodal Chatbot API",
    description="API for audio transcription, facial emotion detection from images/videos, and text sentiment analysis.",
    version="1.0.0",
)
#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_current_user(token: str = Depends(oauth2_scheme), db: DBSession = Depends(get_db)):
    session = db.query(Session).filter(Session.token == token).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou session expirée"
        )

    user = db.query(User).filter(User.id == session.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur introuvable"
        )

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

#app.mount("/static", StaticFiles(directory="C:/Users/user/Downloads/fastapi-suit-project-main-v6/site", html=True), name="static")
app.mount("/static", StaticFiles(directory="/Users/meriemhamdane/Downloads/fastapi-suit-project-main/site", html=True), name="static")

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
async def register_user(data: RegisterRequest, db: DBSession = Depends(get_db)):
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
async def login(data: LoginRequest, db: DBSession = Depends(get_db)):
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

    return {"message": "Connexion réussie", "token": new_session.token, "session_id": new_session.id}

# --- Global Model Loading (for performance) ---

# Whisper Model (loaded on demand for 'base' model, as it's efficient enough)
WHISPER_MODEL = "base"
WHISPER_MODEL_GLOBAL = None # Initialized to None, loaded in startup

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
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
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

# --- NEW: XGBoost Audio Emotion Model Globals ---
XGB_AUDIO_EMOTION_MODEL_PATH = "models/xgb_model.joblib" # Make sure this path is correct!
XGB_AUDIO_EMOTION_MODEL = None
# Étiquettes émotions (ordre important) - Match your sample
XGB_AUDIO_EMOTION_LABELS = ['neutral', 'calm', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise']


# --- Model and Tokenizer Initialization in startup ---
@app.on_event("startup")
async def load_models_on_startup():
    global text_sentiment_tokenizer, text_sentiment_model, TEXT_SENTIMENT_LABELS
    global text_emotion_tokenizer, text_emotion_model, TEXT_EMOTION_LABELS # No global for TEXT_EMOTION_LABELS if manually defined
    global XGB_AUDIO_EMOTION_MODEL # NEW: Load XGBoost model
    global WHISPER_MODEL_GLOBAL
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
            #TEXT_SENTIMENT_LABELS = [label for _, label in label_map]
            TEXT_SENTIMENT_LABELS = ["negative", "neutral", "positive"] 
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


    # Load Whisper Model
    try:
        logger.info(f"Loading Whisper ASR model '{WHISPER_MODEL}'...")
        WHISPER_MODEL_GLOBAL = whisper.load_model(WHISPER_MODEL)
        logger.info("Whisper ASR model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Whisper ASR model: {e}. Ensure Whisper is installed and models are available.", exc_info=True)
        WHISPER_MODEL_GLOBAL = None


    # NEW: Load XGBoost Audio Emotion Model
    try:
        logger.info(f"Loading XGBoost audio emotion model from '{XGB_AUDIO_EMOTION_MODEL_PATH}'...")
        XGB_AUDIO_EMOTION_MODEL = joblib.load(XGB_AUDIO_EMOTION_MODEL_PATH)
        logger.info("XGBoost audio emotion model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading XGBoost audio emotion model: {e}. Ensure '{XGB_AUDIO_EMOTION_MODEL_PATH}' exists and joblib is installed.", exc_info=True)
        XGB_AUDIO_EMOTION_MODEL = None

# --- Helper Functions ---


async def transcribe_audio_with_whisper(audio_file_path: str) -> str:
    """
    Transcribes an audio file using Whisper, offloading the CPU work to a background thread.
    """
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = await asyncio.to_thread(model.transcribe, audio_file_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Whisper transcription failed: {e}. Ensure FFmpeg is installed and in PATH."
        )
    
async def transcribe_audio_with_whisper_content(audio_data: bytes) -> str:
    """
    Transcribes audio (or audio from video) bytes using the globally loaded Whisper ASR model.
    """
    global WHISPER_MODEL_GLOBAL
    if WHISPER_MODEL_GLOBAL is None:
        raise HTTPException(status_code=503, detail="Whisper ASR model not loaded. Please check server startup logs.")

    # Use a temporary .webm file since Whisper/FFmpeg can handle various audio/video inputs
    # If the input `audio_data` is video, whisper will use ffmpeg to extract audio.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_file_path = temp_audio_file.name

    try:
        # Use asyncio.to_thread for blocking Whisper transcribe call
        # Ensure 'ffmpeg' is installed for Whisper to process .webm or other video files.
        result = await asyncio.to_thread(WHISPER_MODEL_GLOBAL.transcribe, temp_file_path, language="en", fp16=False)
        transcribed_text = result["text"]
        logger.info(f"Transcription successful: {transcribed_text[:70]}...")
        return transcribed_text
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}", exc_info=True)
        return f"[Transcription Failed: {e}]"
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Removed temporary audio/video file: {temp_file_path}")
    
def convert_webm_to_wav(input_path: str) -> str:
    """
    Converts a webm file to wav using ffmpeg and returns the new file path.
    """
    output_path = input_path.replace(".webm", ".wav")
    try:
        ffmpeg.input(input_path).output(output_path, format='wav').run(overwrite_output=True, quiet=True)
        return output_path
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        raise HTTPException(status_code=500, detail="FFmpeg conversion failed.")



async def convert_to_wav(upload_file: UploadFile) -> str:
    try:
        # Sauvegarder le fichier temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_input:
            tmp_input.write(await upload_file.read())
            tmp_input.flush()

        # Créer un chemin de sortie WAV temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_output:
            tmp_output_path = tmp_output.name

        # Commande FFmpeg pour conversion WebM → WAV
        command = [
            "ffmpeg",
            "-y",                   # overwrite
            "-i", tmp_input.name,
            "-ar", "16000",         # Sample rate
            "-ac", "1",             # Mono
            tmp_output_path
        ]

        result = subprocess.run(
            command,
            capture_output=True
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"FFmpeg conversion failed: {result.stderr.decode()}"
            )

        return tmp_output_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
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
        logger.warning("Text sentiment analysis model not loaded or labels not defined. Returning default.")
        return SentimentAnalysisResult(
            overall_sentiment="N/A",
            confidence_score=0.0,
            raw_scores={},
            message="Text sentiment model not available."
        )

    if not text.strip():
        return SentimentAnalysisResult(
            overall_sentiment="N/A",
            confidence_score=0.0,
            raw_scores={},
            message="No text to analyze sentiment."
        )

    try:
        device = get_device()
        inputs = text_sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run model inference in a thread to avoid blocking
        outputs = await asyncio.to_thread(text_sentiment_model, **inputs)
        
        # Softmax to get probabilities
        probabilities = torch.softmax(outputs.logits, dim=1)[0] # Get probabilities for the single batch item

        predicted_index = torch.argmax(probabilities).item()
        predicted_label = TEXT_SENTIMENT_LABELS[predicted_index]
        confidence_score = probabilities[predicted_index].item()

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
        logger.error(f"Detailed text sentiment analysis error with local GPT-2 model: {e}", exc_info=True)
        return SentimentAnalysisResult(
            overall_sentiment="Error",
            confidence_score=0.0,
            raw_scores={},
            message=f"Text sentiment analysis failed: {e}"
        )

async def analyze_text_emotion_logic(text: str) -> TextEmotionAnalysisResult:
    """
    First analyzes sentiment of the text, then concatenates the sentiment result
    with the original text, and finally analyzes emotions of this combined text
    using the locally loaded Hugging Face GPT-2 emotion model.
    """
    if text_emotion_model is None or text_emotion_tokenizer is None or not TEXT_EMOTION_LABELS:
        raise HTTPException(status_code=503, detail="Text emotion analysis model not loaded or labels not defined. Check server logs.")

    # --- Step 1: Perform Sentiment Analysis on the original text ---
    try:
        sentiment_result = await analyze_text_sentiment_logic(text)
        overall_sentiment_label = sentiment_result.overall_sentiment
        # Adjust label mapping if needed based on your sentiment model's output
        if overall_sentiment_label == "LABEL_2" or overall_sentiment_label == "positive":
            overall_sentiment_label = "positive"
        elif overall_sentiment_label == "LABEL_0" or overall_sentiment_label == "negative":
            overall_sentiment_label = "negative"
        else: # Handle "LABEL_1" or "neutral"
            overall_sentiment_label = "neutral"
            
        logger.info(f"Sentiment for emotion input text '{text[:50]}...': {overall_sentiment_label}")
    except HTTPException as e:
        # If sentiment analysis fails, log and propagate the error
        logger.error(f"Failed to perform sentiment analysis before emotion analysis: {e.detail}")
        raise HTTPException(status_code=500, detail=f"Dependency error: Sentiment analysis failed before emotion analysis: {e.detail}")
    except Exception as e:
        logger.error(f"Unexpected error during sentiment analysis for emotion input: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during sentiment analysis: {e}")


    # --- Step 2: Concatenate sentiment result with the original text ---
    # Format: "Sentiment: [sentiment_label]. [original_text]"
    combined_text_for_emotion = f"Sentiment: {overall_sentiment_label}. {text}"
    logger.info(f"Combined text for emotion analysis: '{combined_text_for_emotion[:100]}...'")

    # --- Step 3: Perform Emotion Analysis on the combined text ---
    try:
        device = get_device()
        inputs = text_emotion_tokenizer(combined_text_for_emotion, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = await asyncio.to_thread(text_emotion_model, **inputs) # Run in thread to avoid blocking
        
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

        predicted_index = torch.argmax(probabilities).item()
        # Use the manually defined labels (or model's id2label if it has human-readable ones)
        predicted_label = TEXT_EMOTION_LABELS[predicted_index] 
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
        logger.error(f"Detailed text emotion analysis error with local GPT-2 model on combined text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text emotion analysis failed: {e}")


# NEW: Function to extract MFCC features
def extract_mfcc_from_audio_data(audio_data_bytes: bytes, sr: int = 16000, n_mfcc: int = 40):
    """
    Extracts mean MFCC features from raw audio bytes (or audio extracted from video).
    Uses a temporary file as librosa.load typically works with file paths.
    """
    try:
        # Create a temporary file to save the audio bytes for librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio_file:
            temp_audio_file.write(audio_data_bytes)
            temp_file_path = temp_audio_file.name

        # Load audio data using librosa from the temporary file.
        # librosa.load can directly read audio from video files if ffmpeg is installed.
        y, current_sr = librosa.load(temp_file_path, sr=None)

        # If the sample rate is not 16000, resample it
        if current_sr != sr:
            y = librosa.resample(y=y, orig_sr=current_sr, target_sr=sr)
            logger.info(f"Resampled audio from {current_sr} Hz to {sr} Hz for MFCC extraction.")

        # Ensure y is float32 (librosa usually returns float32)
        y = y.astype(np.float32)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0) # Take mean across time frames
        return mfcc_mean

    except Exception as e:
        logger.error(f"Error extracting MFCC from audio data: {e}", exc_info=True)
        raise ValueError(f"MFCC extraction failed: {e}")
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# NEW: Function to analyze audio emotion using XGBoost model
async def analyze_audio_emotion_xgboost(audio_bytes: bytes) -> XGBoostAudioEmotionResult:
    """
    Analyzes emotion from audio bytes (or audio extracted from video bytes) using the globally loaded XGBoost model.
    """
    global XGB_AUDIO_EMOTION_MODEL, XGB_AUDIO_EMOTION_LABELS

    if XGB_AUDIO_EMOTION_MODEL is None:
        return XGBoostAudioEmotionResult(
            dominant_emotion=None,
            confidence=0.0,
            all_scores={},
            message="XGBoost audio emotion model not loaded. Please check server startup logs."
        )

    try:
        # Extract MFCC features - this can be a blocking operation, run in threadpool
        features_np = await asyncio.to_thread(extract_mfcc_from_audio_data, audio_bytes)

        # Reshape for model prediction
        features_reshaped = features_np.reshape(1, -1)

        # Predict emotion probabilities - this is a blocking operation
        proba = await asyncio.to_thread(XGB_AUDIO_EMOTION_MODEL.predict_proba, features_reshaped)
        proba = proba[0] # Get probabilities for the single prediction

        predicted_index = np.argmax(proba).item()
        predicted_label = XGB_AUDIO_EMOTION_LABELS[predicted_index]
        confidence_score = proba[predicted_index].item()

        raw_scores = {
            XGB_AUDIO_EMOTION_LABELS[i]: float(p)
            for i, p in enumerate(proba)
        }

        return XGBoostAudioEmotionResult(
            dominant_emotion=predicted_label,
            confidence=confidence_score,
            all_scores=raw_scores,
            message="XGBoost audio emotion analysis successful."
        )

    except ValueError as ve: # Catch specific ValueError from MFCC extraction
        logger.error(f"XGBoost audio emotion analysis failed due to MFCC extraction: {ve}")
        return XGBoostAudioEmotionResult(
            dominant_emotion=None, confidence=0.0, all_scores={}, message=f"MFCC extraction error: {ve}"
        )
    except Exception as e:
        logger.error(f"XGBoost audio emotion analysis failed: {e}", exc_info=True)
        return XGBoostAudioEmotionResult(
            dominant_emotion=None, confidence=0.0, all_scores={}, message=f"Prediction error: {e}"
        )

# --- API Endpoints ---

# --- Audio Transcription Endpoint ---


@app.post("/transcribe", response_model=TranscriptionResult, summary="Transcribe Audio File")
async def transcribe_audio(audio: UploadFile = File(..., description="Audio file to transcribe")):
    """
    Transcribes an uploaded audio file into text using the local Whisper model.
    """
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are supported.")

    suffix = f".{audio.filename.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio_file:
        content = await audio.read()
        temp_audio_file.write(content)
        temp_file_path = temp_audio_file.name

    converted_file_path = None
    try:
        if suffix == ".webm":
            converted_file_path = convert_webm_to_wav(temp_file_path)
            transcription = await transcribe_audio_with_whisper(converted_file_path)
        else:
            transcription = await transcribe_audio_with_whisper(temp_file_path)

        return TranscriptionResult(text=transcription)
    finally:
        os.unlink(temp_file_path)
        if converted_file_path and os.path.exists(converted_file_path):
            os.unlink(converted_file_path)


@app.post("/transcribe_real_time")
async def transcribe_real_time(file: UploadFile = File(...)):
    try:
        wav_path = await convert_to_wav(file)
        logger.info(f"Fichier converti en WAV: {wav_path}")
        transcription = await transcribe_audio_with_whisper(wav_path)
        logger.info(f"Transcription obtenue: {transcription}")
        return JSONResponse(content={"text": transcription})
    except Exception as e:
        logger.error(f"Erreur transcribe_real_time: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})




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


# --- Video Analysis Endpoint ---
@app.post("/analyze-video-emotions", response_model=VideoAnalysisResult, summary="Analyze Facial Emotions and Sentiments from Video")
async def analyze_video_emotions_endpoint(
    video: UploadFile = File(..., description="Video file to analyze facial emotions and sentiments")
):
    """
    Processes an uploaded video file to extract and analyze facial emotions and sentiments frame by frame.
    """
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only video files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{video.filename.split('.')[-1]}") as temp_video_file:
        content = await video.read()
        temp_video_file.write(content)
        temp_file_path = temp_video_file.name

    try:
        analysis_results = await process_video_for_emotions(temp_file_path)
        return analysis_results
    except Exception as e:
        logger.error(f"Video analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


# --- Text Sentiment Analysis Endpoint ---
@app.post("/analyze-text-sentiment", response_model=SentimentAnalysisResult, summary="Analyze Text Sentiment")
async def analyze_text_sentiment_endpoint(text: str = Form(..., description="Text to analyze sentiment")):
    """
    Analyzes the sentiment of provided text (e.g., positive, neutral, negative).
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    return await analyze_text_sentiment_logic(text)

# --- Text Emotion Analysis Endpoint ---
@app.post("/analyze-text-emotion", response_model=TextEmotionAnalysisResult, summary="Analyze Text Emotion")
async def analyze_text_emotion_endpoint(text: str = Form(..., description="Text to analyze emotion")):
    """
    Analyzes the emotion of provided text (e.g., anger, joy, sadness).
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    return await analyze_text_emotion_logic(text)

# NEW HTTP Endpoint for XGBoost Audio Emotion Analysis
@app.post("/audio-emotion-analysis-xgboost", response_model=XGBoostAudioEmotionResult, summary="Analyze Audio Emotion using XGBoost")
async def audio_emotion_analysis_xgboost_endpoint(audio_file: UploadFile = File(..., description="Audio file to analyze emotion")):
    """
    Analyzes emotion from an uploaded audio file using the XGBoost model.
    """
    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are supported.")
    
    audio_bytes = await audio_file.read()
    
    try:
        result = await analyze_audio_emotion_xgboost(audio_bytes)
        return result
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions from the analysis function
    except Exception as e:
        logger.error(f"Error processing audio for XGBoost emotion analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze audio emotion: {e}")



# --- AGGREGATION LOGIC (COPIED DIRECTLY INTO MAIN.PY) ---

def aggregate_predictions(
    probs_text : Optional[Dict[str, float]],
    probs_audio: Optional[Dict[str, float]],
    probs_image: Optional[Dict[str, float]], # Renamed from probs_video for clarity w.r.t facial emotions
    task: str= "sentiment",
    f1_text : float = 0.95, # Placeholder: Replace with actual F1-score of your text model
    f1_audio: float = 0.42, # Placeholder: Replace with actual F1-score of your audio model
    f1_image: float = 0.84, # Placeholder: Replace with actual F1-score of your image/facial model
) -> Dict[str, float]:
    """
    Agrège les distributions de probas provenant de texte, audio et image.

    Args
    ----
    probs_* : dict(label -> proba) ou None si modalité absente.
    task    : 'sentiment' ou 'emotion'.
    f1_* : F1-score macro de chaque modèle (poids de confiance).

    Returns
    -------
    dict avec proba agrégées, 'final_label' et 'confidence'.
    """

    # 1) jeu de labels selon la tâche
    if task.lower() == "sentiment":
        # Ensure these labels match what your sentiment models (text, image, audio if applicable) output
        labels = ['positive', 'neutral', 'negative'] 
    elif task.lower() == "emotion":
        # Ensure these labels match what your emotion models (audio, image, text if applicable) output
        labels = ['anger', 'disgust', 'fear', 'joy',
                  'sadness', 'surprise', 'neutral'] # Note: 'calm' from XGB is currently excluded to align with 7-emotion set
                                                     # If 'calm' is critical, you need to align all models to 8 emotions.
    else:
        raise ValueError("task doit être 'sentiment' ou 'emotion'")

    # 2) collecter modalités disponibles et leurs poids
    modalities: List[Dict[str, float]] = []
    weights    : List[float]          = []

    # Text modality
    if probs_text is not None:
        # It's crucial that probs_text contains *only* the labels for the current task.
        # E.g., if task is 'sentiment', probs_text should only have 'positive', 'neutral', 'negative'
        # If your text model outputs more (e.g., emotions), you need to filter or pass the correct subset.
        modalities.append(probs_text)
        weights.append(f1_text)

    # Audio modality
    if probs_audio is not None:
        # Filter audio probabilities to match the current task's labels
        filtered_audio_probs = {k: v for k, v in probs_audio.items() if k in labels}
        if filtered_audio_probs: # Only add if there are relevant probabilities after filtering
            modalities.append(filtered_audio_probs)
            weights.append(f1_audio)
        else:
            logger.warning(f"Audio modality has no relevant labels for '{task}' task after filtering.")


    # Image/Facial modality
    if probs_image is not None:
        # Filter image probabilities to match the current task's labels
        filtered_image_probs = {k: v for k, v in probs_image.items() if k in labels}
        if filtered_image_probs: # Only add if there are relevant probabilities after filtering
            modalities.append(filtered_image_probs)
            weights.append(f1_image)
        else:
            logger.warning(f"Image modality has no relevant labels for '{task}' task after filtering.")


    if not modalities:
        logger.warning(f"No modalities provided or relevant for aggregation task '{task}'.")
        return {
            'final_label': 'N/A',
            'confidence': 0.0,
            'raw_scores': {label: 0.0 for label in labels} if labels else {},
            'message': 'Aucune modalité fournie ou pertinente pour l\'agrégation.'
        }

    weights = np.asarray(weights, dtype=float)
    total   = weights.sum()
    weights = weights / total if total else weights   # évite la division par 0

    logger.info(f"Aggregation weights for {task}: {weights}")

    # 3) moyenne pondérée
    agg = {lab: 0.0 for lab in labels}
    for w, probs in zip(weights, modalities):
        for lab in labels:
            # Use .get(lab, 0.0) in case a modality doesn't have a specific label
            agg[lab] += w * probs.get(lab, 0.0)

    # Normalize aggregated probabilities so they sum to 1
    sum_agg_probs = sum(agg[lab] for lab in labels)
    if sum_agg_probs > 0:
        agg = {lab: prob / sum_agg_probs for lab, prob in agg.items()}
    else:
        # Handle case where all aggregated probabilities are zero
        # Default to 'neutral' for both sentiment and emotion if no signal
        default_label = 'neutral'
        if default_label in labels:
            agg = {lab: 0.0 for lab in labels} # Reset all to 0.0
            agg[default_label] = 1.0 # Give default 100%
        else: # Fallback if 'neutral' is not in the labels for some reason
            if labels:
                agg = {lab: 1/len(labels) for lab in labels} # Distribute equally
            else:
                agg = {} # Empty if no labels

    # 4) label final
    if not agg: # Should not happen if labels are defined, but as a safeguard
        return {
            'final_label': 'N/A',
            'confidence': 0.0,
            'raw_scores': {},
            'message': 'Erreur interne lors de l\'agrégation des probabilités (agrégation vide).'
        }
    
    # Ensure all labels in `labels` list are in `agg` dict for consistency
    for lab in labels:
        if lab not in agg:
            agg[lab] = 0.0

    best = max(agg, key=agg.get)
    agg_result = {
        'final_label': best,
        'confidence': agg[best],
        'raw_scores': agg # Include all scores for transparency
    }
    return agg_result


# --- Helper to aggregate video (facial) emotions into a single probability distribution ---
def aggregate_facial_emotions_to_overall(
    emotions_timeline: List[FrameEmotionDetail],
    emotion_labels: List[str] # EMOTION_LABELS from your facial model
) -> Dict[str, float]:
    """
    Aggregates frame-level facial emotion probabilities into an overall video emotion probability.
    Averages probabilities across all detected faces and frames.
    """
    total_probs = {label: 0.0 for label in emotion_labels}
    face_count = 0

    for frame in emotions_timeline:
        for face in frame.detected_faces:
            face_count += 1
            for label, prob in face.emotions.items():
                if label in total_probs: # Ensure the label is one we are tracking
                    total_probs[label] += prob

    if face_count == 0:
        return {label: 0.0 for label in emotion_labels} # No faces detected
    
    # Average the probabilities
    averaged_probs = {label: prob_sum / face_count for label, prob_sum in total_probs.items()}
    return averaged_probs

# --- Helper to aggregate video (facial) sentiments into a single probability distribution ---
def aggregate_facial_sentiments_to_overall(
    emotions_timeline: List[FrameEmotionDetail], # 'emotions_timeline' contains both emotions and sentiments
    sentiment_labels: List[str] # SENTIMENT_IMAGE_LABELS from your facial sentiment model
) -> Dict[str, float]:
    """
    Aggregates frame-level facial sentiment probabilities into an overall video sentiment probability.
    Averages probabilities across all detected faces and frames.
    """
    total_probs = {label: 0.0 for label in sentiment_labels}
    face_count = 0

    for frame in emotions_timeline:
        for face in frame.detected_faces:
            if face.sentiments: # Ensure sentiments are present for this face
                face_count += 1
                for label, prob in face.sentiments.items():
                    if label in total_probs: # Ensure the label is one we are tracking
                        total_probs[label] += prob

    if face_count == 0:
        return {label: 0.0 for label in sentiment_labels} # No faces detected
    
    # Average the probabilities
    averaged_probs = {label: prob_sum / face_count for label, prob_sum in total_probs.items()}
    return averaged_probs



# --- Main Endpoint with Multimodal Fusion ---
@app.post("/analyze-interview-video", response_model=InterviewAnalysisResult) # Add response_model for clarity
async def analyze_interview_video(
    video: UploadFile = File(...)
):
    temp_file_path = None
    try:
        # Save the uploaded video temporarily for all analyses that need the file path
        content = await video.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{video.filename}") as temp_video_file:
            temp_video_file.write(content)
            temp_file_path = temp_video_file.name

        logger.info(f"Received video: {video.filename}, size: {len(content)} bytes. Saved to {temp_file_path}")

        # Run individual analysis tasks concurrently
        transcription_task = transcribe_audio_with_whisper_content(content) # Use content directly
        video_analysis_task = process_video_for_emotions(temp_file_path) # Needs file path for CV2
        xgb_audio_emotion_task = analyze_audio_emotion_xgboost(content) # Use content directly
        text_sentiment_task = analyze_text_sentiment_logic(None) # Initialize, will run after transcription
        text_emotion_task = analyze_text_emotion_logic(None) # Initialize, will run after transcription


        # Await initial results (transcription, video facial analysis, audio emotion)
        transcription_text, video_emotions_result, xgb_audio_emotion_result = await asyncio.gather(
            transcription_task,
            video_analysis_task,
            xgb_audio_emotion_task
        )
        logger.info("Individual analysis (transcription, video, audio emotion) tasks completed.")

        # Now run text sentiment and emotion based on transcription_text
        text_sentiment_result = await analyze_text_sentiment_logic(transcription_text)
        text_emotion_result = await analyze_text_emotion_logic(transcription_text) # Get text emotion raw scores

        logger.info(f"Text sentiment analysis complete: {text_sentiment_result.overall_sentiment}")
        logger.info(f"Text emotion analysis complete: {text_emotion_result.overall_emotion}")


        # --- Aggregate Facial Emotion and Sentiment from VideoAnalysisResult ---
        # Facial emotion aggregation (for 'image' modality in aggregate_predictions for EMOTION task)
        overall_facial_emotions_probs = aggregate_facial_emotions_to_overall(
            video_emotions_result.emotions_timeline,
            emotion_labels=EMOTION_LABELS # Use your facial emotion model's labels
        )
        logger.info(f"Overall facial emotion probabilities: {overall_facial_emotions_probs}")

        # Facial sentiment aggregation (for 'image' modality in aggregate_predictions for SENTIMENT task)
        overall_facial_sentiments_probs = aggregate_facial_sentiments_to_overall(
            video_emotions_result.emotions_timeline, # The same timeline contains sentiment data
            sentiment_labels=SENTIMENT_IMAGE_LABELS # Use your facial sentiment model's labels
        )
        logger.info(f"Overall facial sentiment probabilities: {overall_facial_sentiments_probs}")


        # --- MULTIMODAL FUSION ---

        # 1. Aggregate Sentiment
        # Important: Pass raw_scores, and align labels with aggregate_predictions' sentiment labels
        aggregated_sentiment_raw = aggregate_predictions(
            probs_text=text_sentiment_result.raw_scores,
            probs_audio=xgb_audio_emotion_result.all_scores, # Assuming XGB audio model also implicitly gives sentiment or you'd pass None
            probs_image=overall_facial_sentiments_probs,
            task="sentiment",
            # Replace with actual macro F1-scores of your models
            f1_text=0.95, # Example F1 for Text Sentiment Model
            f1_audio=0.42, # Example F1 for Audio Sentiment (if applicable)
            f1_image=0.84  # Example F1 for Facial Sentiment Model
        )
        aggregated_sentiment = SentimentAnalysisResult(
            overall_sentiment=aggregated_sentiment_raw.get('final_label', 'N/A'),
            confidence_score=aggregated_sentiment_raw.get('confidence', 0.0),
            raw_scores=aggregated_sentiment_raw.get('raw_scores', {})
        )
        logger.info(f"Multimodal Sentiment: {aggregated_sentiment.overall_sentiment} (Conf: {aggregated_sentiment.confidence_score:.2f})")


        # 2. Aggregate Emotion
        # Important: Pass raw_scores, and align labels with aggregate_predictions' emotion labels
        aggregated_emotion_raw = aggregate_predictions(
            probs_text=text_emotion_result.raw_scores, # Use text emotion model's raw scores
            probs_audio=xgb_audio_emotion_result.all_scores,
            probs_image=overall_facial_emotions_probs,
            task="emotion",
            # Replace with actual macro F1-scores of your models
            f1_text=0.90, # Example F1 for Text Emotion Model
            f1_audio=0.55, # Example F1 for Audio Emotion Model
            f1_image=0.80  # Example F1 for Facial Emotion Model
        )
        # Reusing XGBoostAudioEmotionResult for multimodal_emotion
        aggregated_emotion = XGBoostAudioEmotionResult(
            dominant_emotion=aggregated_emotion_raw.get('final_label', 'N/A'),
            confidence=aggregated_emotion_raw.get('confidence', 0.0),
            all_scores=aggregated_emotion_raw.get('raw_scores', {})
        )
        logger.info(f"Multimodal Emotion: {aggregated_emotion.dominant_emotion} (Conf: {aggregated_emotion.confidence:.2f})")


        # --- Construct Final Response ---
        response_data = InterviewAnalysisResult(
            transcription=TranscriptionResult(text=transcription_text),
            video_emotions=video_emotions_result,
            overall_text_sentiment=text_sentiment_result,
            xgb_audio_emotion=xgb_audio_emotion_result,
            multimodal_sentiment=aggregated_sentiment,
            multimodal_emotion=aggregated_emotion
        )
        
        return JSONResponse(content=response_data.model_dump()) # Use model_dump() for Pydantic v2

    except HTTPException as e:
        logger.error(f"HTTPException during video analysis for {video.filename}: {e.detail}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Unhandled error during full interview video analysis for {video.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to analyze interview video due to an internal server error: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Removed temporary video file: {temp_file_path}")


# --- NEW: WebSocket Endpoint for Real-time Webcam Analysis ---

@app.websocket("/ws/analyze-webcam")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established for real-time webcam analysis.")

    try:
        while True:
            image_bytes = await websocket.receive_bytes()

            # Analyse émotions et sentiments en parallèle (ou séquentiellement)
            emotions = await analyze_facial_emotions(image_bytes)
            sentiments = await analyze_facial_sentiments(image_bytes)

            # Association par bounding box (simple fusion basée sur les coordonnées)
            combined_results = []

            for emo_face in emotions:
                bbox = emo_face["bounding_box"]
                matched_sentiment = next(
                    (s for s in sentiments if s["bounding_box"] == bbox),
                    None
                )

                combined_results.append({
                    "bounding_box": bbox,
                    "emotions": emo_face.get("emotions", {}),
                    "sentiments": matched_sentiment.get("sentiments", {}) if matched_sentiment else {}
                })

            response_data = {
                "facial_emotions": combined_results
            }

            await websocket.send_json(response_data)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected from client.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "error": str(e),
            "message": "An error occurred during real-time analysis."
        })
    finally:
        logger.info("WebSocket connection closed.")



# route principale POST
@app.post("/api/save-webcam-analysis")
async def save_webcam_analysis_api(request: Request, db: Session = Depends(get_db)):
    print("✅ Appel API reçu !")
    data = await request.json()
    
    print("📦 Données reçues :", data)
    # Extraction des données JSON
    session_id = data.get("session_id")
    dominant_emotion = data.get("dominant_emotion")
    dominant_sentiment = data.get("dominant_sentiment")
    video_emotions = data.get("video_emotions")
    duration = data.get("duration_seconds")
    frames = data.get("frames")

    # Appel à ta fonction de sauvegarde
    save_facial_analysis(
        db, session_id,
        dominant_emotion,
        dominant_sentiment, video_emotions,
        duration, frames
    )

    return {"message": "Facial analysis saved successfully."}

@app.post("/api/save-facial-emotions")
async def save_facial_emotions(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    # extraire et sauvegarder les émotions
    session_id = data.get("session_id")
    dominant_emotion = data.get("dominant_emotion")
    dominant_sentiment = data.get("dominant_sentiment")
    video_emotions = data.get("video_emotions")
    duration = data.get("duration_seconds")
    frames = data.get("frames")
    save_facial_analysis(
        db, session_id,
        dominant_emotion,
        dominant_sentiment, video_emotions,
        duration, frames
    )
    return {"message": "Facial emotions saved."}

@app.post("/api/save-transcription")
async def save_transcription_api(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    session_id = data["session_id"]
    trans = data["transcription"]
    if isinstance(trans, (dict, list)):
        trans = json.dumps(trans)
    save_transcription(db,session_id ,trans)
    return {"message": "Transcription saved."}

@app.post("/api/save-text-sentiment")
async def save_text_sentiment_api(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    session_id = data["session_id"]
    sentiment_label = data["sentiment_label"]
    confidence_score = data["confidence_score"]
    raw_scores = data["raw_scores"]

    # Si raw_scores est une string JSON, on le convertit en dict
    if isinstance(raw_scores, str):
        try:
            raw_scores = json.loads(raw_scores)
        except json.JSONDecodeError:
            raw_scores = {}

    save_text_sentiment(db, session_id, sentiment_label, confidence_score, raw_scores)
    return {"message": "Text sentiment analysis saved."}

'''@app.get("/api/user-sessions")
def get_user_sessions(
    limit: int = 5,
    current_user: UserCredentials = Depends(get_current_user),
    db: DBSession = Depends(get_db)
):
    user_id = current_user.id

    # Appel de ta fonction utilitaire (on suppose qu'elle trie déjà par date décroissante si souhaité)
    all_sessions = get_sessions_by_user_id(user_id, db)

    result = []

    for sess in all_sessions:
        fa = sess.facial_analyses[0] if sess.facial_analyses else None
        ts = sess.text_sentiments[0] if sess.text_sentiments else None

        # On skippe les sessions sans AUCUNE analyse
        if not fa and not ts:
            continue

        result.append({
            "session_id": sess.id,
            "created_at": sess.created_at.isoformat(),
            "dominant_emotion": fa.dominant_emotion if fa else None,
            "dominant_sentiment_facial": fa.dominant_sentiment if fa else None,
            "video_emotions": fa.video_emotions if fa else None,
            "dominant_sentiment_text": ts.sentiment_label if ts else None,
            "text_confidence": ts.confidence_score if ts else None
        })

        # Stopper si on a atteint la limite demandée
        if len(result) >= limit:
            break

    return result'''

@app.get("/api/user-sessions")
def get_user_sessions(
    limit: int = 5,
    current_user: UserCredentials = Depends(get_current_user),
    db: DBSession = Depends(get_db)
):
    user_id = current_user.id

    all_sessions = get_sessions_by_user_id(user_id, db)

    result = []

    for sess in all_sessions:
        # Filtrer les analyses faciales non vides
        valid_facial_analyses = [
            {
                "dominant_emotion": fa.dominant_emotion,
                "dominant_sentiment_facial": fa.dominant_sentiment,
                "video_emotions": fa.video_emotions
            }
            for fa in sess.facial_analyses
            if fa.dominant_emotion or fa.dominant_sentiment or fa.video_emotions
        ]

        # Filtrer les analyses textuelles non vides
        valid_text_sentiments = [
            {
                "dominant_sentiment_text": ts.sentiment_label,
                "text_confidence": ts.confidence_score
            }
            for ts in sess.text_sentiments
            if ts.sentiment_label
        ]

        # Skip si aucune analyse valable
        if not valid_facial_analyses and not valid_text_sentiments:
            continue

        result.append({
            "session_id": sess.id,
            "created_at": sess.created_at.isoformat(),
            "facial_analyses": valid_facial_analyses,
            "text_sentiments": valid_text_sentiments
        })

        if len(result) >= limit:
            break

    return result

@app.get("/api/user-info")
def get_user_info(db: DBSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    return {
        "full_name": f"{current_user.first_name} {current_user.last_name}",
        "target_position": current_user.target_position,
        "session_count": len(current_user.sessions)
    }


@app.get("/api/dashboard-metrics")
def get_dashboard_metrics(db: DBSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    user_sessions = current_user.sessions

    total_sessions = len(user_sessions)
    this_week = datetime.utcnow() - timedelta(days=7)
    sessions_this_week = db.query(Session).filter(
        Session.user_id == current_user.id,
        Session.created_at >= this_week
    ).count()

    # Durée totale
    total_duration_seconds = db.query(func.sum(FacialAnalysis.duration_seconds)).join(Session).filter(
        Session.user_id == current_user.id
    ).scalar() or 0
    total_duration_hours = round(total_duration_seconds / 3600, 1)

    '''duration_this_week_seconds = db.query(func.sum(FacialAnalysis.duration_seconds)).join(Session).filter(
        Session.user_id == current_user.id,
        Session.created_at >= this_week
    ).scalar() or 0
    duration_this_week_hours = round(duration_this_week_seconds / 3600, 1)'''
    total_duration_seconds = db.query(func.sum(FacialAnalysis.duration_seconds)).join(Session).filter(
    Session.user_id == current_user.id
    ).scalar() or 0

    total_duration_hours = round(total_duration_seconds / 3600, 3)

    # Sentiments positifs
    '''positive_sentiments = db.query(TextSentiment).join(Session).filter(
        Session.user_id == current_user.id,
        TextSentiment.sentiment_label == "positive"
    ).all()
    positive_ratio = round((len(positive_sentiments) / max(1, total_sessions)) * 100)'''
    positive_sentiments = db.query(
    func.avg(cast(func.JSON_EXTRACT(TextSentiment.raw_scores, '$.positive'), Float))
    ).join(Session).filter(
    Session.user_id == current_user.id
    ).scalar()
    positive_ratio = round((positive_sentiments or 0) * 100)


    # Émotions dominantes
    last_emotion = db.query(FacialAnalysis.dominant_emotion).join(Session).filter(
        Session.user_id == current_user.id
    ).order_by(desc(Session.created_at)).first()
    last_emotion = last_emotion[0] if last_emotion else "N/A"

    previous_emotion = db.query(FacialAnalysis.dominant_emotion).join(Session).filter(
        Session.user_id == current_user.id,
        Session.created_at < this_week
    ).order_by(desc(Session.created_at)).first()
    previous_emotion = previous_emotion[0] if previous_emotion else "N/A"

    return {
        "positive_sentiment": {
            "value": f"{positive_ratio} %",
            "change": f"+ {positive_ratio} this week",  # calcul dynamique possible
        },
        "practice_time": {
            "value": f"{total_duration_hours}h",
            "change": f"+{total_duration_hours}h this week"
        },
        "dominant_emotion": {
            "value": last_emotion,
            "change": f"{previous_emotion} last week"
        },
        "sessions": {
            "value": total_sessions,
            "change": f"+{sessions_this_week} this week"
        }
    }


@app.post("/add-question/", status_code=201)
def create_question(question: QuestionCreate, db: DBSession = Depends(get_db),  current_user: User = Depends(get_current_user)):
    try:
        save_question(db, question_text=question.question_text, user_id=current_user.id)
        return {"message": "Question added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/questions/", response_model=List[QuestionOut])
async def get_questions(db: Session = Depends(get_db)):
    questions = db.query(QuestionsBank).all()
    return questions

def save_stress_analysis(db, user_id: int, data: StressAnalysisCreate):
    db_stress = StressAnalysis(
        user_id=user_id,
        session_id=data.session_id,
        stress_facial=data.stress_facial,
        stress_textuel=data.stress_textuel,
        stress_global=data.stress_global
    )
    db.add(db_stress)
    db.commit()
    db.refresh(db_stress)
    return db_stress

@app.post("/stress_analysis/", response_model=StressAnalysisRead)
def create_stress_analysis(data: StressAnalysisCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_stress = save_stress_analysis(db,current_user.id, data)
    return db_stress

@app.get("/stress_analysis/user/", response_model=List[StressAnalysisRead])
def read_stress_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    records = get_user_stress_history(db, current_user.id)
    if not records:
        raise HTTPException(status_code=404, detail="No stress analysis found for this user")
    return records




@app.get("/analytics/emotions", tags=["Analytics"])
def get_emotion_distribution(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(Session).filter(Session.user_id == current_user.id).all()
    all_emotions = []

    for s in sessions:
        for fa in s.facial_analyses:
            if fa.dominant_emotion:
                all_emotions.append(fa.dominant_emotion)

    emotion_count = dict(Counter(all_emotions))
    return {"labels": list(emotion_count.keys()), "data": list(emotion_count.values())}


@app.get("/analytics/sentiments", tags=["Analytics"])
def get_sentiment_distribution(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(Session).filter(Session.user_id == current_user.id).all()
    sentiments = []

    for s in sessions:
        for ts in s.text_sentiments:
            if ts.sentiment_label:
                sentiments.append(ts.sentiment_label)

    sentiment_count = dict(Counter(sentiments))
    return {"labels": list(sentiment_count.keys()), "data": list(sentiment_count.values())}


@app.get("/analytics/stress-over-time", tags=["Analytics"])
def get_stress_over_time(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    analyses = (
        db.query(StressAnalysis)
        .filter(StressAnalysis.user_id == current_user.id)
        .order_by(StressAnalysis.created_at)
        .all()
    )

    dates = [a.created_at.strftime("%Y-%m-%d") for a in analyses]
    values = [a.stress_global for a in analyses]

    return {"labels": dates, "data": values}


# --- Root Endpoint for API testing ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Multimodal Chatbot API! Visit /docs for API documentation."}