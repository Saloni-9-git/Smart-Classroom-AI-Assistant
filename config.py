from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# folders
DATABASE_DIR = BASE_DIR / "database"
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DB_DIR = BASE_DIR / "vector_store"
LOG_DIR = BASE_DIR / "logs"

# create folders
for folder in [DATABASE_DIR, UPLOAD_DIR, VECTOR_DB_DIR, LOG_DIR]:
    os.makedirs(folder, exist_ok=True)

# database file
DB_PATH = DATABASE_DIR / "classroom.db"

# api key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")