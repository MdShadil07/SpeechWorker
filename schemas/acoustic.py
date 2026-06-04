from pydantic import BaseModel
from typing import List, Optional


class PhoneInterval(BaseModel):
    phoneme: str
    startTime: float
    endTime: float
    durationMs: Optional[float] = None


class AcousticAnalysisRequest(BaseModel):
    audioPath: str
    phones: List[PhoneInterval]


class AcousticPhoneFeatures(BaseModel):
    phoneme: str
    startTime: float
    endTime: float
    durationMs: float
    rmsDb: float
    zeroCrossingRate: float
    spectralCentroid: float
    spectralBandwidth: float
    mfccMean: List[float]
    pitchMean: float
    pitchMax: float
    pitchSlope: float
    confidence: float


class AcousticAnalysisResponse(BaseModel):
    sampleRate: int
    duration: float
    phones: List[AcousticPhoneFeatures]
