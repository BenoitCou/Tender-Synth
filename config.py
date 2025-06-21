"""
Centralised configuration and constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv


DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(DOTENV_PATH, override=True)

# --- API KEYS --------------------------------------------------------------
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
MISTRAL_API_KEY     = os.getenv("MISTRAL_API_KEY", "")

# --- CLIENT DATA ------------------------------------------------------------
CLIENT              = os.getenv("CLIENT", "")
CLIENT_DESCRIPTION  = os.getenv("CLIENT_DESCRIPTION", "")


# --- Path to the controlled vocabulary -------------------------------------
LEXICON_PATH        = os.getenv("LEXICON_PATH", "")

