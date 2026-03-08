import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def _get_groq_api_key() -> str:
    try:
        if "GROQ" in st.secrets and "API_KEY" in st.secrets["GROQ"]:
            return st.secrets["GROQ"]["API_KEY"]
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")

GROQ_API_KEY: str = _get_groq_api_key()

ALLOWED_FILE_TYPES: list[str] = ["pdf", "docx", "txt", "md", "py", "json", "csv"]

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

_IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_SHARING_MODE") == "true" or \
                      os.path.exists("/mount/src")
CHROMA_DB_PATH: str = (
    "/tmp/chroma_db" if _IS_STREAMLIT_CLOUD
    else os.path.join("data", "chroma_db")
)

TOP_K: int = 5

LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS: int = 1024
LLM_TEMPERATURE: float = 0.2