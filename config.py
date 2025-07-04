"""
Centralised configuration and constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

def get_secret(key):
    return st.secrets[key] if key in st.secrets else os.getenv(key)

DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(DOTENV_PATH, override=True)

# --- API KEYS --------------------------------------------------------------
OPENAI_API_KEY      = get_secret("OPENAI_API_KEY")
MISTRAL_API_KEY     = get_secret("MISTRAL_API_KEY")

# --- CLIENT DATA ------------------------------------------------------------
CLIENT              = get_secret("CLIENT")
CLIENT_DESCRIPTION  = get_secret("CLIENT_DESCRIPTION")


# --- Path to the controlled vocabulary -------------------------------------
LEXICON_PATH        = get_secret("LEXICON_PATH")

