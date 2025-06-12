from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError

# Variables séparées
DB_USER = "root"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_NAME = "Suit_DB"

# Construction dynamique de l'URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# Création de l'engine SQLAlchemy
engine = create_engine(DATABASE_URL)

# Session locale
SessionLocal = sessionmaker(bind=engine)

#  Base pour les modèles
Base = declarative_base()

# Test de connexion
try:
    with engine.connect() as connection:
        print("✅ Connexion réussie à la base de données.")
except OperationalError as e:
    print("❌ Erreur de connexion à la base de données :", e)