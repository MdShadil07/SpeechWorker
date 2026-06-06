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

        # GLOBAL PITCH EXTRACTION (Massive Optimization)
        # Running pyin globally once instead of looping per-phoneme drops time from 2+ minutes to 1 second
        global_f0 = None
        global_voiced = None
        hop_length = 512
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                global_f0, global_voiced, _ = librosa.pyin(
                    y,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sr,
                    hop_length=hop_length,
                    fill_na=None
                )
        except Exception as e:
            logger.warn("Global pitch extraction failed", error=str(e))

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

            features.append(self._extract_phone_features(
                phone.phoneme, 
                start_seconds, 
                end_seconds, 
                segment, 
                sr,
                global_f0,
                global_voiced,
                hop_length
            ))

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
        global_f0: np.ndarray = None,
        global_voiced: np.ndarray = None,
        hop_length: int = 512
    ) -> AcousticPhoneFeatures:
        duration_ms = max(0.0, (end_seconds - start_seconds) * 1000)
        rms = float(np.sqrt(np.mean(np.square(segment))) if segment.size else 0.0)
        rms_db = float(20 * math.log10(max(rms, 1e-7)))
        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(segment)))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr)))
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            
        mfcc_mean = [float(value) for value in np.mean(mfcc, axis=1)]

        # Extract pitch (F0) from global arrays
        pitch_mean = 0.0
        pitch_max = 0.0
        pitch_slope = 0.0
        
        if global_f0 is not None and global_voiced is not None:
            # Map time to frames
            start_frame = int((start_seconds * sr) / hop_length)
            end_frame = int((end_seconds * sr) / hop_length)
            
            # Ensure indices are within bounds
            start_frame = max(0, min(len(global_f0), start_frame))
            end_frame = max(start_frame, min(len(global_f0), end_frame))
            
            # For extremely short phonemes, grab at least 1 frame if possible
            if start_frame == end_frame and start_frame < len(global_f0):
                end_frame = start_frame + 1
                
            if end_frame > start_frame:
                f0_slice = global_f0[start_frame:end_frame]
                voiced_slice = global_voiced[start_frame:end_frame]
                
                valid_f0 = f0_slice[voiced_slice]
                if len(valid_f0) > 0:
                    pitch_mean = float(np.mean(valid_f0))
                    pitch_max = float(np.max(valid_f0))
                    
                    # Calculate slope if we have enough points
                    if len(valid_f0) > 1:
                        x = np.arange(len(valid_f0))
                        slope, _ = np.polyfit(x, valid_f0, 1)
                        pitch_slope = float(slope)

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
