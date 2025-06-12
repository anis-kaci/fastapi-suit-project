from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from passlib.hash import bcrypt
from db.models import User, UserCredentials, Session as UserSession
from db.database import SessionLocal
from datetime import datetime

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_user(db: Session, first_name, last_name, birth_date, education, position, email, password):
    user = User(
        first_name=first_name,
        last_name=last_name,
        birth_date=birth_date,
        education_level=education,
        target_position=position
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    credentials = UserCredentials(
        user_id=user.id,
        email=email,
        hashed_password=bcrypt.hash(password)
    )
    db.add(credentials)
    db.commit()
    return user

def authenticate_user(db: Session, email: str, password: str):
    creds = db.query(UserCredentials).filter(UserCredentials.email == email).first()
    if not creds or not bcrypt.verify(password, creds.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return creds.user

def create_session(db: Session, user_id: int):
    new_session = UserSession(user_id=user_id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session
