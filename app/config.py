import os
from typing import Optional

class Config:
    # Whisper model configuration
    # Use int8 by default to reduce memory (float32 uses 4x more memory)
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small.en")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # Server configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8001"))
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*") # Comma-separated list for production

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Audio processing
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1  # mono

config = Config()