# config.py
import os
import streamlit as st

try:
    # First, try Streamlit Cloud secrets
    GROQ_API_KEY = st.secrets["GROQ"]["API_KEY"]
except (KeyError, RuntimeError):
    # Fallback for local development using .env
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Optional: other config constants
ALLOWED_FILE_TYPES = ["txt", "pdf", "docx", "md", "py"]