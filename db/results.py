from db.models import Transcription, FacialAnalysis, TextSentiment, QuestionsBank, StressAnalysis


def save_transcription(db, session_id, text):
    db_transcript = Transcription(session_id=session_id, text=text)
    db.add(db_transcript)
    db.commit()

def save_facial_analysis(db, session_id, dominant_emotion, dominant_sentiment, video_emotions, duration, frames):
    db_analysis = FacialAnalysis(
        session_id=session_id,
        dominant_emotion=dominant_emotion,
        dominant_sentiment=dominant_sentiment,
        video_emotions=video_emotions,
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

def save_question(db, question_text, user_id=None):
    db_question = QuestionsBank(
        question_text=question_text,
        user_id=user_id
    )
    db.add(db_question)
    db.commit()


'''def save_stress_analysis(db, user_id, session_id, stress_facial, stress_textuel, stress_global):
    db_stress = StressAnalysis(
        user_id=user_id,
        session_id=session_id,
        stress_facial=stress_facial,
        stress_textuel=stress_textuel,
        stress_global=stress_global
    )
    db.add(db_stress)
    db.commit()
    db.refresh(db_stress)  # pour retourner l'objet avec ID, timestamp, etc.
    return db_stress'''


