from db.models import Transcription, FacialAnalysis, TextSentiment

def save_transcription(db, session_id, text):
    db_transcript = Transcription(session_id=session_id, text=text)
    db.add(db_transcript)
    db.commit()

def save_facial_analysis(db, session_id, dominant_emotion, emotion_timeline, dominant_sentiment, sentiment_timeline, duration, frames):
    db_analysis = FacialAnalysis(
        session_id=session_id,
        dominant_emotion=dominant_emotion,
        emotion_timeline=emotion_timeline,
        dominant_sentiment=dominant_sentiment,
        sentiment_timeline=sentiment_timeline,
        duration_seconds=duration,
        frames_analyzed=frames
    )
    db.add(db_analysis)
    db.commit()

def save_text_sentiment(db, session_id, label, confidence, raw_scores):
    db_sentiment = TextSentiment(
        session_id=session_id,
        sentiment_label=label,
        confidence_score=confidence,
        raw_scores=raw_scores
    )
    db.add(db_sentiment)
    db.commit()
