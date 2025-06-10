# main.py

import whisper
import tempfile
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
# NEW: Import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

# Imports for Sentiment Analysis
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


# --- Pydantic Models for API Responses ---
class TranscriptionResult(BaseModel):
    text: str

class EmotionDetection(BaseModel):
    bounding_box: dict
    emotion: str
    confidence: float

class FacialAnalysisResult(BaseModel): # Changed to BaseModel
    facial_emotions: list[EmotionDetection]

class FrameEmotionDetail(BaseModel):
    timestamp_seconds: int
    detected_faces: list[EmotionDetection]

class VideoAnalysisResult(BaseModel):
    video_duration_seconds: int
    frames_analyzed: int
    emotions_timeline: list[FrameEmotionDetail]

class SentimentAnalysisResult(BaseModel):
    overall_sentiment: str
    confidence_score: float
    raw_scores: dict

# NEW Pydantic model for combined interview analysis result
class InterviewAnalysisResult(BaseModel):
    transcription: TranscriptionResult
    video_emotions: VideoAnalysisResult
    overall_text_sentiment: SentimentAnalysisResult


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multimodal Chatbot API",
    description="API for audio transcription, facial emotion detection from images/videos, and text sentiment analysis.",
    version="1.0.0",
)

# NEW: Add CORS Middleware
# This allows requests from any origin to your API.
# In a production environment, you would restrict 'allow_origins' to specific domains.
origins = [
    "*" # Allows all origins for local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Global Model Loading (for performance) ---

# ... (rest of your main.py code remains the same) ...

# Whisper Model (loaded on demand for 'base' model, as it's efficient enough)
WHISPER_MODEL = "base"

# Facial Analysis Model (your best_model_so_far.keras)
EMOTION_MODEL_PATH = "best_model_so_far.keras" # Make sure this file is in your project root or provide the full path
EMOTION_MODEL = None
FACE_DETECTOR = None # Initialize face detector

try:
    EMOTION_MODEL = load_model(EMOTION_MODEL_PATH)
    print(f"Loaded facial emotion model: {EMOTION_MODEL_PATH}")

    # Load Haar Cascade Classifier for face detection
    FACE_DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if FACE_DETECTOR.empty():
        print("WARNING: Haar Cascade for face detection not loaded. Facial analysis might fail.")
except Exception as e:
    print(f"Error loading facial emotion model or face detector: {e}. Ensure 'best_model_so_far.keras' exists and TensorFlow/OpenCV are installed.")
    EMOTION_MODEL = None
    FACE_DETECTOR = None

# Define facial emotion labels and input size globally, based on your model
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
MODEL_INPUT_SIZE = (96, 96) # Height, Width


# --- Sentiment Analysis Model (Pre-trained from Hugging Face Hub) ---
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest" 

sentiment_pipeline = None
SENTIMENT_LABELS = [] # Will be populated upon successful model load

try:
    sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_ID, tokenizer=SENTIMENT_MODEL_ID)
    
    if hasattr(sentiment_pipeline.model.config, 'id2label'):
        label_map = sorted(sentiment_pipeline.model.config.id2label.items(), key=lambda x: int(x[0]))
        SENTIMENT_LABELS = [label for _, label in label_map]
    else:
        SENTIMENT_LABELS = ["negative", "neutral", "positive"] 
        print(f"WARNING: Could not auto-detect sentiment labels from model config. Using default: {SENTIMENT_LABELS}")
    
    print(f"Loaded sentiment model from Hugging Face Hub: {SENTIMENT_MODEL_ID} with labels: {SENTIMENT_LABELS}")

except Exception as e:
    print(f"Error loading sentiment model from Hugging Face Hub ({SENTIMENT_MODEL_ID}): {e}. "
          "Ensure 'transformers' is installed and your internet connection is active for initial download.")
    sentiment_pipeline = None
    SENTIMENT_LABELS = []


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


async def analyze_facial_emotions(image_bytes: bytes) -> list[EmotionDetection]:
    """
    Analyzes facial emotions from a single image's bytes.
    """
    if EMOTION_MODEL is None or FACE_DETECTOR is None:
        raise HTTPException(status_code=500, detail="Facial emotion model or detector not loaded. Check server logs.")

    try:
        np_image = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image file.")

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = FACE_DETECTOR.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_emotions = []
        if len(faces) == 0:
            return []

        for (x, y, w, h) in faces:
            face_rgb_roi = rgb_img[y:y+h, x:x+w]

            if face_rgb_roi.size == 0:
                continue

            try:
                face_input = preprocess_face_for_model(face_rgb_roi)
                predictions = EMOTION_MODEL.predict(face_input, verbose=0)[0]

                predicted_emotion_index = np.argmax(predictions)
                predicted_emotion = EMOTION_LABELS[predicted_emotion_index]
                confidence = float(predictions[predicted_emotion_index])

                detected_emotions.append(EmotionDetection(
                    bounding_box={"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    emotion=predicted_emotion,
                    confidence=confidence
                ))
            except Exception as e:
                print(f"Error predicting emotion for a face: {e}")
                continue

        return detected_emotions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Facial analysis internal error: {e}")


async def process_video_for_emotions(video_file_path: str) -> VideoAnalysisResult:
    """
    Processes a video file frame by frame (sampling one per second) to analyze facial emotions.
    """
    if EMOTION_MODEL is None or FACE_DETECTOR is None:
        raise HTTPException(status_code=500, detail="Facial emotion model or detector not loaded.")

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file. Ensure FFmpeg is installed and the video file is valid.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = int(total_frames / fps) if fps > 0 else 0

    emotions_timeline = []
    frames_processed = 0

    for sec in range(video_duration_seconds + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()

        if not ret:
            if sec < video_duration_seconds:
                 print(f"Warning: Could not read frame at {sec} seconds. Skipping.")
            continue

        frames_processed += 1

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = FACE_DETECTOR.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_emotions_for_frame = []
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_rgb_roi = rgb_img[y:y+h, x:x+w]

                if face_rgb_roi.size == 0:
                    continue

                try:
                    face_input = preprocess_face_for_model(face_rgb_roi)
                    predictions = EMOTION_MODEL.predict(face_input, verbose=0)[0]

                    predicted_emotion_index = np.argmax(predictions)
                    predicted_emotion = EMOTION_LABELS[predicted_emotion_index]
                    confidence = float(predictions[predicted_emotion_index])

                    detected_emotions_for_frame.append(EmotionDetection(
                        bounding_box={"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                        emotion=predicted_emotion,
                        confidence=confidence
                    ))
                except Exception as e:
                    print(f"Error predicting emotion for a face in frame {sec}s: {e}")
                    continue
        
        emotions_timeline.append(FrameEmotionDetail(
            timestamp_seconds=sec,
            detected_faces=detected_emotions_for_frame
        ))

    cap.release()
    return VideoAnalysisResult(
        video_duration_seconds=video_duration_seconds,
        frames_analyzed=frames_processed,
        emotions_timeline=emotions_timeline
    )


async def analyze_sentiment_with_hf_pipeline(text: str) -> SentimentAnalysisResult:
    """
    Analyzes sentiment of text using the locally loaded Hugging Face pipeline.
    """
    if sentiment_pipeline is None or not SENTIMENT_LABELS:
        raise HTTPException(status_code=500, detail="Sentiment analysis pipeline not loaded or labels not defined. Check server logs.")

    try:
        result = sentiment_pipeline(text)
        
        main_prediction = result[0]
        predicted_label = main_prediction['label']
        confidence_score = main_prediction['score']

        raw_scores = {}
        inputs = sentiment_pipeline.tokenizer(text, return_tensors="pt", truncation=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        sentiment_pipeline.model.to(device)

        with torch.no_grad():
            outputs = sentiment_pipeline.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        
        for i, prob in enumerate(probabilities):
            if i < len(SENTIMENT_LABELS):
                raw_scores[SENTIMENT_LABELS[i]] = prob.item()
            else:
                raw_scores[f"LABEL_{i}"] = prob.item()


        return SentimentAnalysisResult(
            overall_sentiment=predicted_label,
            confidence_score=confidence_score,
            raw_scores=raw_scores
        )

    except Exception as e:
        print(f"Detailed sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")


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
@app.post("/analyze-face", response_model=FacialAnalysisResult, summary="Analyze Facial Emotions from Image")
async def analyze_face(image: UploadFile = File(..., description="Image file to analyze facial emotions")):
    """
    Detects faces in an uploaded image and predicts their emotions using a custom CNN model.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are supported (e.g., JPEG, PNG).")

    image_bytes = await image.read()
    facial_emotions = await analyze_facial_emotions(image_bytes)

    if not facial_emotions:
        return FacialAnalysisResult(facial_emotions=[])
    
    return FacialAnalysisResult(facial_emotions=facial_emotions)

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
    Analyzes the sentiment of provided text using a pre-trained model from Hugging Face Hub.
    Returns the overall sentiment, its confidence score, and raw scores for all possible labels.
    """
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for sentiment analysis.")
    
    sentiment_result = await analyze_sentiment_with_hf_pipeline(text)
    return sentiment_result

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
        print(f"Transcription complete: {transcription_text[:50]}...") # Log first 50 chars

        # Step 2: Perform Facial Analysis from Video
        video_emotions_result = await process_video_for_emotions(temp_file_path)
        print(f"Facial analysis complete for {video_emotions_result.frames_analyzed} frames.")

        # Step 3: Analyze Sentiment of the Transcribed Text
        overall_text_sentiment_result = await analyze_sentiment_with_hf_pipeline(transcription_text)
        print(f"Text sentiment analysis complete: {overall_text_sentiment_result.overall_sentiment}")

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
        print(f"Error during full interview video analysis for {video.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze interview video: {e}")
    finally:
        # Clean up the temporary video file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"Cleaned up temporary video file: {temp_file_path}")


# --- Root Endpoint for API testing ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Multimodal Chatbot API! Visit /docs for API documentation."}