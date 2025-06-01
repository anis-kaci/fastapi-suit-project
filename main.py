import os
import tempfile
import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import whisper
import requests # Keep requests if you need it for other things, but it won't be used for sentiment
from dotenv import load_dotenv

# Import the necessary components from transformers
from transformers import pipeline

# Load environment variables (still useful if you have other keys, but HUGGINGFACE_API_KEY won't be mandatory for sentiment)
load_dotenv()

app = FastAPI(
    title="Audio Transcription and Sentiment Analysis API",
    description="API for transcribing audio files using local Whisper and performing sentiment analysis with Hugging Face's RoBERTa model."
)

# --- Configuration ---
# HUGGINGFACE_API_KEY is now optional for sentiment, but keep it if you need it for other HF API calls
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") 
# You can remove the ValueError if you don't need the HF API for anything else.
# If you keep it, make sure the .env file is present or API calls will fail.

WHISPER_MODEL = "base"
ROBERTA_MODEL = "joeddav/xlm-roberta-large-xnli" # Use the model name from your notebook

# Initialize the sentiment analysis pipeline globally to avoid re-loading on every request
# This will download the model the first time the server starts.
try:
    sentiment_classifier = pipeline("zero-shot-classification", model=ROBERTA_MODEL)
    print(f"Loaded local sentiment model: {ROBERTA_MODEL}")
except Exception as e:
    print(f"Error loading local sentiment model {ROBERTA_MODEL}: {e}")
    print("Sentiment analysis will not be available.")
    sentiment_classifier = None # Set to None if loading fails

# --- Models (Pydantic) ---
# ... (These remain the same as before) ...
class SentimentAnalysisResult(BaseModel):
    overallSentiment: str = "neutral"
    confidenceLevel: int = 5
    emotionalTone: str = "neutral"
    score: int = 50
    sentimentScores: dict = {}
    emotionalScores: dict = {}
    detailedAnalysis: dict

class CombinedAnalysisResult(BaseModel):
    transcription: str
    timestamp: str
    analysis: SentimentAnalysisResult
    rawAnalysis: dict

# --- Helper Functions ---

async def transcribe_audio_with_whisper(audio_file_path: str) -> str:
    # ... (This function remains the same as before) ...
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(audio_file_path)
        return result["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whisper transcription failed: {e}. "
                                                     f"Ensure Whisper is installed and dependencies (like torch) are met.")

async def analyze_with_roberta(text: str) -> dict:
    """
    Analyzes sentiment of text using the locally loaded RoBERTa model.
    """
    if sentiment_classifier is None:
        raise HTTPException(status_code=500, detail="Sentiment analysis model not loaded.")

    candidate_labels = [
        "positive", "negative", "neutral", 
        "confident", "nervous", "enthusiastic", 
        "professional", "anxious", "calm", "excited"
    ]
    
    try:
        # Use the locally loaded pipeline
        result = sentiment_classifier(text, candidate_labels=candidate_labels)
        return result[0] # The pipeline returns a list, we need the first (and only) result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local RoBERTa analysis failed: {e}")


def process_roberta_analysis(roberta_data: dict) -> SentimentAnalysisResult:
    # ... (This function remains the same as before) ...
    if not roberta_data or not roberta_data.get('labels') or not roberta_data.get('scores'):
        return SentimentAnalysisResult(
            overallSentiment="neutral",
            confidenceLevel=5,
            emotionalTone="neutral",
            score=50,
            detailedAnalysis=roberta_data
        )

    labels = roberta_data['labels']
    scores = roberta_data['scores']

    overall_sentiment = "neutral"
    highest_sentiment_score = 0
    sentiment_scores_dict = {"positive": 0, "negative": 0, "neutral": 0}

    sentiment_labels = ["positive", "negative", "neutral"]
    for sentiment in sentiment_labels:
        try:
            index = labels.index(sentiment)
            sentiment_scores_dict[sentiment] = scores[index]
            if scores[index] > highest_sentiment_score:
                highest_sentiment_score = scores[index]
                overall_sentiment = sentiment
        except ValueError:
            pass # Label not found, score remains 0

    emotional_tone = "neutral"
    highest_emotional_score = 0
    emotional_scores_dict = {}

    emotional_labels = ["confident", "nervous", "enthusiastic", "professional", "anxious", "calm", "excited"]
    for emotion in emotional_labels:
        try:
            index = labels.index(emotion)
            emotional_scores_dict[emotion] = scores[index]
            if scores[index] > highest_emotional_score:
                highest_emotional_score = scores[index]
                emotional_tone = emotion
        except ValueError:
            pass # Label not found, score remains 0

    confidence_level = round(highest_sentiment_score * 10)
    score = round(highest_sentiment_score * 100)

    return SentimentAnalysisResult(
        overallSentiment=overall_sentiment,
        confidenceLevel=confidence_level,
        emotionalTone=emotional_tone,
        score=score,
        sentimentScores=sentiment_scores_dict,
        emotionalScores=emotional_scores_dict,
        detailedAnalysis=roberta_data
    )

# --- Endpoints (remain the same, but now use the local classifier) ---
# ...
@app.post("/transcribe", summary="Transcribe Audio File")
async def transcribe_audio(audio: UploadFile = File(...)):
    # ... (remains the same) ...
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.filename.split('.')[-1]}") as temp_audio_file:
        content = await audio.read()
        temp_audio_file.write(content)
        temp_file_path = temp_audio_file.name

    try:
        transcription = await transcribe_audio_with_whisper(temp_file_path)
        return {"text": transcription}
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.post("/sentiment", response_model=SentimentAnalysisResult, summary="Analyze Sentiment of Text")
async def sentiment_analysis(text: str = Form(...)):
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for sentiment analysis.")

    # Call local sentiment analysis
    roberta_analysis = await analyze_with_roberta(text)
    processed_analysis = process_roberta_analysis(roberta_analysis)
    return processed_analysis

@app.post("/advanced-sentiment", summary="Get Raw Sentiment Analysis Data")
async def advanced_sentiment_analysis(text: str = Form(...)):
    if not text:
        raise HTTPException(status_code=400, detail="No text provided for advanced sentiment analysis.")

    # Call local sentiment analysis
    roberta_analysis = await analyze_with_roberta(text)
    return roberta_analysis

@app.post("/analyze", response_model=CombinedAnalysisResult, summary="Combined Audio Transcription and Sentiment Analysis")
async def combined_analysis(audio: UploadFile = File(None), text: str = Form(None)):
    transcribed_text = None
    temp_file_path = None

    if audio:
        if not audio.content_type or not audio.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are supported.")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.filename.split('.')[-1]}") as temp_audio_file:
            content = await audio.read()
            temp_audio_file.write(content)
            temp_file_path = temp_audio_file.name
        
        try:
            transcribed_text = await transcribe_audio_with_whisper(temp_file_path)
        finally:
            if temp_file_path:
                os.unlink(temp_file_path)
        
    final_text_for_analysis = transcribed_text if transcribed_text else text

    if not final_text_for_analysis:
        raise HTTPException(status_code=400, detail="No audio file or text provided for analysis.")

    roberta_analysis_raw = {}
    processed_analysis_result = SentimentAnalysisResult()

    try:
        # Call local sentiment analysis
        roberta_analysis_raw = await analyze_with_roberta(final_text_for_analysis)
        processed_analysis_result = process_roberta_analysis(roberta_analysis_raw)
    except HTTPException as e:
        print(f"Sentiment analysis failed for text '{final_text_for_analysis[:50]}...': {e.detail}")
        processed_analysis_result = SentimentAnalysisResult(
            overallSentiment="neutral",
            confidenceLevel=0,
            emotionalTone="unknown",
            score=0,
            detailedAnalysis={"error": f"Sentiment analysis could not be completed: {e.detail}"}
        )

    return CombinedAnalysisResult(
        transcription=final_text_for_analysis,
        timestamp=datetime.datetime.now().isoformat(),
        analysis=processed_analysis_result,
        rawAnalysis=roberta_analysis_raw
    )