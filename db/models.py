'''from sqlalchemy import Column, Integer, String, Date, ForeignKey, Float, DateTime, Text, JSON, func
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    birth_date = Column(Date, nullable=False)
    education_level = Column(String(100))
    target_position = Column(String(100))
    credentials = relationship("UserCredentials", back_populates="user", uselist=False)
    sessions = relationship("Session", back_populates="user")
    # One-to-many : un user a plusieurs analyses de stress
    stress_analyses = relationship("StressAnalysis", back_populates="user", cascade="all, delete-orphan")


class UserCredentials(Base):
    __tablename__ = 'user_credentials'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    user = relationship("User", back_populates="credentials")

class Session(Base):
    __tablename__ = 'sessions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Nouveau champ token, unique et indexé pour faciliter les recherches
    token = Column(String, unique=True, index=True, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="sessions")
    transcriptions = relationship("Transcription", back_populates="session")
    facial_analyses = relationship("FacialAnalysis", back_populates="session")
    text_sentiments = relationship("TextSentiment", back_populates="session")

class Transcription(Base):
    __tablename__ = 'transcriptions'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id', ondelete='CASCADE'), nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="transcriptions")


class FacialAnalysis(Base):
    __tablename__ = 'facial_analyses'

    id = Column(Integer, primary_key=True)  # Assumes id is manually set unless AUTO_INCREMENT is added
    session_id = Column(Integer, ForeignKey('sessions.id', ondelete='CASCADE'), nullable=False)
    dominant_emotion = Column(String(50))
    dominant_sentiment = Column(String(50))
    video_emotions = Column(JSON)  # JSON string stored as text
    duration_seconds = Column(Float)
    frames_analyzed = Column(Integer)

    session = relationship("Session", back_populates="facial_analyses")


class TextSentiment(Base):
    __tablename__ = 'text_sentiments'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id', ondelete='CASCADE'), nullable=False)
    sentiment_label = Column(String(50))
    confidence_score = Column(Float)
    raw_scores = Column(JSON)
    session = relationship("Session", back_populates="text_sentiments")


class QuestionsBank(Base):
    __tablename__ = 'questions_bank'
    question_id = Column(Integer, primary_key=True, index=True)
    question_text = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)



class StressAnalysis(Base):
    __tablename__ = "stress_analysis"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)

    stress_facial = Column(Float)
    stress_textuel = Column(Float)
    stress_global = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations si besoin (optionnel mais utile si tu fais du join)
    user = relationship("User", back_populates="stress_analyses")
    session = relationship("Session", back_populates="stress_analyses")'''

from sqlalchemy import Column, Integer, String, Date, ForeignKey, Float, DateTime, Text, JSON, func
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    birth_date = Column(Date, nullable=False)
    education_level = Column(String(100))
    target_position = Column(String(100))
    credentials = relationship("UserCredentials", back_populates="user", uselist=False)
    sessions = relationship("Session", back_populates="user")
    # Un user a plusieurs analyses de stress
    stress_analyses = relationship("StressAnalysis", back_populates="user", cascade="all, delete-orphan")


class UserCredentials(Base):
    __tablename__ = 'user_credentials'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    user = relationship("User", back_populates="credentials")


class Session(Base):
    __tablename__ = 'sessions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    token = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="sessions")
    transcriptions = relationship("Transcription", back_populates="session")
    facial_analyses = relationship("FacialAnalysis", back_populates="session")
    text_sentiments = relationship("TextSentiment", back_populates="session")

    # AJOUT IMPORTANT : relation avec StressAnalysis
    stress_analyses = relationship("StressAnalysis", back_populates="session", cascade="all, delete-orphan")


class Transcription(Base):
    __tablename__ = 'transcriptions'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id', ondelete='CASCADE'), nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    session = relationship("Session", back_populates="transcriptions")


class FacialAnalysis(Base):
    __tablename__ = 'facial_analyses'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id', ondelete='CASCADE'), nullable=False)
    dominant_emotion = Column(String(50))
    dominant_sentiment = Column(String(50))
    video_emotions = Column(JSON)
    duration_seconds = Column(Float)
    frames_analyzed = Column(Integer)
    session = relationship("Session", back_populates="facial_analyses")


class TextSentiment(Base):
    __tablename__ = 'text_sentiments'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id', ondelete='CASCADE'), nullable=False)
    sentiment_label = Column(String(50))
    confidence_score = Column(Float)
    raw_scores = Column(JSON)
    session = relationship("Session", back_populates="text_sentiments")


class QuestionsBank(Base):
    __tablename__ = 'questions_bank'
    question_id = Column(Integer, primary_key=True, index=True)
    question_text = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)


class StressAnalysis(Base):
    __tablename__ = "stress_analysis"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)

    stress_facial = Column(Float)
    stress_textuel = Column(Float)
    stress_global = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="stress_analyses")
    session = relationship("Session", back_populates="stress_analyses")
