from sqlalchemy.orm import Session as DBSession, selectinload
from db.models import Session as SessionModel, QuestionsBank, StressAnalysis

'''def get_sessions_by_user_id(user_id: int, db: DBSession):
    """
    Récupère toutes les sessions appartenant à un utilisateur donné.
    """
    return (
        db.query(SessionModel)
        .filter(SessionModel.user_id == user_id)
        .order_by(SessionModel.created_at.desc())
        .all()
    )'''


def get_sessions_by_user_id(user_id: int, db: DBSession):
    """
    Récupère toutes les sessions appartenant à un utilisateur donné,
    en préchargeant les relations nécessaires (analyses faciales et sentiments texte).
    """
    return (
        db.query(SessionModel)
        .options(
            selectinload(SessionModel.facial_analyses),
            selectinload(SessionModel.text_sentiments)
        )
        .filter(SessionModel.user_id == user_id)
        .order_by(SessionModel.created_at.desc())
        .all()
    )
    

def get_user_questions(db: DBSession, user_id: int):
    return db.query(QuestionsBank).filter(
        (QuestionsBank.user_id == None) | (QuestionsBank.user_id == user_id)
    ).all()




def get_user_stress_history(db: DBSession, user_id: int):
    """
    Récupère toutes les analyses de stress d'un utilisateur donné,
    triées par date de création décroissante.
    """
    return (
        db.query(StressAnalysis)
        .options(
            selectinload(StressAnalysis.session),  # si besoin d'infos de la session
            selectinload(StressAnalysis.user)      # si besoin d'infos utilisateur
        )
        .filter(StressAnalysis.user_id == user_id)
        .order_by(StressAnalysis.created_at.desc())
        .all()
    )
