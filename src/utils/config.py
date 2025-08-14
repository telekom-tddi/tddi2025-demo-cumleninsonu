import logging
import os
from pathlib import Path
from dotenv import load_dotenv

_BASE_DIR_FOR_DOTENV = Path(__file__).resolve().parent.parent.parent
dotenv_path = _BASE_DIR_FOR_DOTENV / '.env'

# Load environment variables from .env file, overriding existing ones if present
# and making sure to specify the path.
logger = logging.getLogger(__name__)
logger.info(f"Attempting to load .env file from: {dotenv_path}")
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f".env file loaded successfully from {dotenv_path}.")
else:
    logger.warning(f".env file not found at {dotenv_path}. Environment variables might not be set as expected.")

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent



# LLM settings
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# Call Center settings
CALL_CENTER_MODE = True  # Flag to enable call center mode

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Client request timeout settings (used by webapp when calling API)
API_CONNECT_TIMEOUT = int(os.getenv("API_CONNECT_TIMEOUT", "5"))
API_READ_TIMEOUT = int(os.getenv("API_READ_TIMEOUT", "120"))

# Web app settings
WEBAPP_HOST = os.getenv("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(os.getenv("WEBAPP_PORT", "8501"))
APP_TITLE = "Arama Merkezi Asistanı"
APP_DESCRIPTION = "AI destekli çağrı merkezi asistanı, müşteri hizmetleri ve destek için."

# Cache settings
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
MAX_CACHE_SIZE = 100  # Number of queries to cache
API_USAGE = os.getenv("API_USAGE", "False") == "True"
print(API_USAGE)
# Hugging Face cache settings
# Set environment variables for better caching behavior
os.environ.setdefault("HF_HUB_CACHE", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)
os.environ.setdefault("HF_HOME", CACHE_DIR)
