import math
import numpy as np
import librosa
import structlog
from schemas.acoustic import (
    AcousticAnalysisRequest,
    AcousticAnalysisResponse,
    AcousticPhoneFeatures,
)
from app.config import config


logger = structlog.get_logger()


class AcousticAnalysisService:
    async def analyze(self, request: AcousticAnalysisRequest) -> AcousticAnalysisResponse:
        y, sr = librosa.load(
            request.audioPath,
            sr=config.AUDIO_SAMPLE_RATE,
            mono=True,
        )
        duration = len(y) / sr if sr else 0
        features = []

        for phone in request.phones:
            start_seconds = self._to_seconds(phone.startTime)
            end_seconds = self._to_seconds(phone.endTime)
            start_sample = max(0, min(len(y), int(start_seconds * sr)))
            end_sample = max(start_sample, min(len(y), int(end_seconds * sr)))
            segment = y[start_sample:end_sample]

            if segment.size == 0:
                logger.warn(
                    "Empty acoustic phone segment",
                    phoneme=phone.phoneme,
                    startTime=phone.startTime,
                    endTime=phone.endTime,
                )
                segment = np.zeros(max(1, int(0.01 * sr)), dtype=np.float32)

            features.append(self._extract_phone_features(phone.phoneme, start_seconds, end_seconds, segment, sr))

        return AcousticAnalysisResponse(
            sampleRate=sr,
            duration=duration,
            phones=features,
        )

    def _extract_phone_features(
        self,
        phoneme: str,
        start_seconds: float,
        end_seconds: float,
        segment: np.ndarray,
        sr: int,
    ) -> AcousticPhoneFeatures:
        duration_ms = max(0.0, (end_seconds - start_seconds) * 1000)
        rms = float(np.sqrt(np.mean(np.square(segment))) if segment.size else 0.0)
        rms_db = float(20 * math.log10(max(rms, 1e-7)))
        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(segment)))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr)))
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_mean = [float(value) for value in np.mean(mfcc, axis=1)]

        # Extract pitch (F0) using pyin
        pitch_mean = 0.0
        pitch_max = 0.0
        pitch_slope = 0.0
        
        try:
            # librosa.pyin requires the signal to be at least as long as frame_length.
            # Default frame_length is 2048. For very short phonemes, this fails.
            frame_length = min(2048, max(256, len(segment) - (len(segment) % 2)))
            if len(segment) >= 256:
                f0, voiced_flag, _ = librosa.pyin(
                    segment,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr,
                    frame_length=frame_length,
                    fill_na=None
                )
                valid_f0 = f0[voiced_flag] if f0 is not None and voiced_flag is not None else []
                pitch_mean = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
                pitch_max = float(np.max(valid_f0)) if len(valid_f0) > 0 else 0.0
                
                # Calculate slope if we have enough points
                if len(valid_f0) > 1:
                    x = np.arange(len(valid_f0))
                    slope, _ = np.polyfit(x, valid_f0, 1)
                    pitch_slope = float(slope)
        except Exception as e:
            # If pitch extraction fails (usually due to length), we just leave pitches at 0.0
            pass

        confidence = self._estimate_segment_confidence(
            duration_ms=duration_ms,
            rms_db=rms_db,
            spectral_centroid=spectral_centroid,
        )

        return AcousticPhoneFeatures(
            phoneme=phoneme.upper(),
            startTime=round(start_seconds * 1000, 3),
            endTime=round(end_seconds * 1000, 3),
            durationMs=round(duration_ms, 3),
            rmsDb=round(rms_db, 3),
            zeroCrossingRate=round(zero_crossing_rate, 6),
            spectralCentroid=round(spectral_centroid, 3),
            spectralBandwidth=round(spectral_bandwidth, 3),
            mfccMean=[round(value, 4) for value in mfcc_mean],
            pitchMean=round(pitch_mean, 3),
            pitchMax=round(pitch_max, 3),
            pitchSlope=round(pitch_slope, 3),
            confidence=confidence,
        )

    def _estimate_segment_confidence(
        self,
        duration_ms: float,
        rms_db: float,
        spectral_centroid: float,
    ) -> float:
        duration_score = min(1.0, max(0.0, duration_ms / 80.0))
        energy_score = min(1.0, max(0.0, (rms_db + 55.0) / 35.0))
        spectral_score = 0.35 if spectral_centroid <= 50 else 1.0
        return round(max(0.0, min(1.0, duration_score * 0.35 + energy_score * 0.45 + spectral_score * 0.20)), 3)

    def _to_seconds(self, value: float) -> float:
        # Backend alignment intervals are milliseconds; accept seconds defensively too.
        return value / 1000 if value > 100 else value
