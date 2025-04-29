# backend2/utils/config.py
import os
from typing import Dict, Any


class Config:
    """Configuration for the UI Navigator"""

    # API Keys
    # Changed from MISTRAL_API_KEY
    HF_API_KEY = os.environ.get("HF_API_KEY", "")

    # Model Settings
    LLM_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    LLM_PROVIDER = "nebius"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Vector Store Settings
    VECTOR_DIMENSION = 384
    VECTOR_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  "data", "vector_db")

    # Confidence Thresholds
    MIN_CONFIDENCE = 0.2
    AUTO_EXECUTE_THRESHOLD = 0.7
