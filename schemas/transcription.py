from pydantic import BaseModel, ConfigDict
from typing import List, Optional

class TranscriptionRequest(BaseModel):
    audioPath: str

class Segment(BaseModel):
    start: float
    end: float
    text: str
    confidence: Optional[float] = None

class Word(BaseModel):
    word: str
    start: float
    end: float
    confidence: Optional[float] = None

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    confidence: float
    duration: float
    segments: List[Segment]
    words: List[Word] = []

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    device: str