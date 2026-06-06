import time
# pyrefly: ignore [missing-import]
import structlog
# pyrefly: ignore [missing-import]
import psutil
import os
# pyrefly: ignore [missing-import]
import imageio_ffmpeg
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import librosa
# pyrefly: ignore [missing-import]
from faster_whisper import WhisperModel
from app.config import config
from schemas.transcription import TranscriptionResponse, Segment, Word

# Optimize for AWS m7i-flex.large (2 vCPUs, 8GB RAM)
os.environ['OMP_NUM_THREADS'] = '2'  # Utilize both vCPUs for OpenMP
os.environ['MKL_NUM_THREADS'] = '2'  # Utilize both vCPUs for MKL
os.environ['NUMEXPR_NUM_THREADS'] = '2'  # Utilize both vCPUs for NumExpr
os.environ['MKL_THREADING_LAYER'] = 'GNU'  # Use GNU threading instead of MKL
os.environ['NUMEXPR_MAX_THREADS'] = '2'  # Max threads for NumExpr

logger = structlog.get_logger()


def _ensure_ffmpeg_on_path():
    ffmpeg_path = os.getenv('FFMPEG_PATH') or imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    current_path = os.environ.get('PATH', '')

    if ffmpeg_dir and ffmpeg_dir not in current_path:
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path

    try:
        import audioread.ffdec as ffdec
        ffdec.COMMANDS = (ffmpeg_path, 'ffmpeg', 'avconv')
    except Exception:
        # If audioread is unavailable, librosa may still succeed through soundfile.
        pass

class TranscriptionService:
    def __init__(self):
        self.model = None
        self._load_model()
        self.memory_threshold_percent = 85  # Warn if memory usage exceeds this

    def _get_memory_info(self):
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        memory_percent = process.memory_percent()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'percent': memory_percent
        }

    def _analyze_audio_quality(self, audio_path: str) -> dict:
        """Analyze audio file for quality metrics before transcription"""
        try:
            _ensure_ffmpeg_on_path()

            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Calculate RMS energy (volume)
            rms = np.sqrt(np.mean(y**2))
            rms_db = 20 * np.log10(np.maximum(rms, 1e-5))
            
            # Calculate peak-to-average ratio (dynamic range)
            peak = np.max(np.abs(y))
            peak_db = 20 * np.log10(np.maximum(peak, 1e-5))
            
            # Zero crossing rate (speech vs noise indicator)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Spectral centroid (speech quality indicator)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # Duration
            duration = len(y) / sr
            
            return {
                'rms_db': float(rms_db),
                'peak_db': float(peak_db),
                'zero_crossing_rate': float(zcr),
                'spectral_centroid': float(spectral_centroid),
                'duration_seconds': float(duration),
                'sample_rate': sr,
                'quality_check': {
                    'is_too_quiet': rms_db < -40,  # Very quiet audio
                    'is_clipping': peak_db > -1,    # Audio clipping/distortion
                    'has_speech_content': zcr > 0.05,  # Likely has speech
                    'is_adequate_duration': duration >= 1.0  # At least 1 second
                }
            }
        except Exception as e:
            logger.warn("Audio quality analysis failed", error=str(e))
            return {
                'error': str(e),
                'quality_check': {}
            }

    def _load_model(self):
        """Load Whisper model once at startup with fallback options"""
        models_to_try = [
            (config.WHISPER_MODEL, config.WHISPER_COMPUTE_TYPE),
            ("tiny.en", "int8"),  # Fallback to tiny if configured model fails
        ]
        
        for model_name, compute_type in models_to_try:
            try:
                logger.info(
                    "Loading Whisper model",
                    model=model_name,
                    device=config.WHISPER_DEVICE,
                    compute_type=compute_type
                )
                mem_before = self._get_memory_info()
                start_time = time.time()
                
                self.model = WhisperModel(
                    model_name,
                    device=config.WHISPER_DEVICE,
                    compute_type=compute_type,
                    cpu_threads=2,  # Leverage both vCPUs on m7i-flex.large
                    num_workers=1   # Keep 1 worker to avoid memory duplication
                )
                
                load_time = time.time() - start_time
                mem_after = self._get_memory_info()
                logger.info(
                    "Whisper model loaded successfully",
                    model=model_name,
                    load_time=load_time,
                    mem_before=mem_before,
                    mem_after=mem_after
                )
                return
                
            except RuntimeError as e:
                error_msg = str(e)
                logger.error(
                    "Failed to load Whisper model",
                    model=model_name,
                    error=error_msg,
                    memory_info=self._get_memory_info()
                )
                
                if "mkl_malloc" in error_msg.lower():
                    logger.warn(
                        "Memory allocation failure - will try smaller model",
                        attempted_model=model_name
                    )
                    continue
                else:
                    raise
        
        # If we get here, all fallbacks failed
        raise RuntimeError(
            f"Failed to load any Whisper model. Last attempt: {models_to_try[-1][0]}"
        )

    async def transcribe(self, audio_path: str) -> TranscriptionResponse:
        """Transcribe audio file using faster-whisper"""
        if not self.model:
            raise RuntimeError("Whisper model not loaded")

        start_time = time.time()
        mem_before = self._get_memory_info()

        try:
            logger.info(
                "🎙️ TRANSCRIPTION START",
                audio_path=audio_path,
                memory_before_mb=f"{mem_before['rss_mb']:.1f}MB",
                memory_percent=f"{mem_before['percent']:.1f}%"
            )

            # Stage 1: Load audio file
            load_start = time.time()
            logger.info("⏳ Stage 1: Loading audio file...")

            # Analyze audio quality before transcription
            audio_quality = self._analyze_audio_quality(audio_path)
            audio_load_time = time.time() - load_start

            logger.info(
                "✅ Audio quality analyzed",
                duration_seconds=audio_quality.get('duration_seconds', 0),
                rms_db=audio_quality.get('rms_db', 0),
                peak_db=audio_quality.get('peak_db', 0),
                zero_crossing_rate=audio_quality.get('zero_crossing_rate', 0),
                spectral_centroid=audio_quality.get('spectral_centroid', 0),
                quality_check=audio_quality.get('quality_check', {}),
                load_time_seconds=f"{audio_load_time:.3f}s"
            )

            # Check memory before transcription
            if mem_before['percent'] > self.memory_threshold_percent:
                logger.warn(
                    "⚠️ Memory usage high before transcription",
                    memory_percent=f"{mem_before['percent']:.1f}%",
                    threshold=f"{self.memory_threshold_percent}%"
                )

            # Stage 2: Transcribe audio
            transcribe_start = time.time()
            logger.info(
                "⏳ Stage 2: Running Whisper transcription model...",
                model_config={
                    "beam_size": 5,
                    "patience": 1.0,
                    "compression_ratio_threshold": 2.4,
                    "no_speech_threshold": 0.4,
                    "condition_on_previous_text": False,
                    "vad_filter": True
                }
            )

            segments, info = self.model.transcribe(
                audio_path,
                language="en",
                beam_size=5,
                patience=1.0,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.4,
                condition_on_previous_text=False,
                vad_filter=True,
                without_timestamps=False,
                word_timestamps=True,
            )

            transcribe_time = time.time() - transcribe_start
            logger.info(
                "✅ Whisper transcription complete",
                transcribe_time_seconds=f"{transcribe_time:.3f}s"
            )

            # Stage 3: Process segments and extract words
            segment_start = time.time()
            logger.info("⏳ Stage 3: Processing segments and word alignment...")

            segment_list = []
            word_list = []
            full_text = ""
            total_segments = 0

            for segment_idx, segment in enumerate(segments):
                total_segments += 1
                segment_text = segment.text if hasattr(segment, 'text') else ''
                segment_start_time = segment.start if hasattr(segment, 'start') else 0
                segment_end_time = segment.end if hasattr(segment, 'end') else 0
                segment_confidence = getattr(segment, 'confidence', None)

                logger.info(
                    f"  📝 Segment {segment_idx + 1}: {segment_start_time:.2f}s - {segment_end_time:.2f}s",
                    text=segment_text[:100],  # First 100 chars for readability
                    confidence=segment_confidence
                )

                segment_list.append(Segment(
                    start=segment_start_time,
                    end=segment_end_time,
                    text=segment_text,
                    confidence=segment_confidence
                ))

                # Extract words from segment
                segment_words = 0
                if hasattr(segment, 'words') and getattr(segment, 'words', None):
                    for word_idx, word in enumerate(getattr(segment, 'words', [])):
                        segment_words += 1
                        word_text = getattr(word, 'word', '')
                        word_start = getattr(word, 'start', 0)
                        word_end = getattr(word, 'end', 0)
                        word_confidence = getattr(word, 'confidence', segment_confidence)

                        word_list.append(Word(
                            word=word_text,
                            start=word_start,
                            end=word_end,
                            confidence=word_confidence
                        ))

                        if word_idx < 3 or word_idx % 5 == 0:
                            logger.debug(
                                f"    🔤 Word {word_idx + 1}",
                                word=word_text,
                                start_s=f"{word_start:.3f}",
                                end_s=f"{word_end:.3f}",
                                duration_ms=f"{(word_end - word_start) * 1000:.1f}",
                                confidence=word_confidence
                            )
                else:
                    # Fallback: estimate word timings
                    tokens = segment_text.strip().split()
                    if tokens:
                        duration = max(0.001, segment_end_time - segment_start_time) if segment_end_time > segment_start_time else 0.1 * len(tokens)
                        token_duration = duration / len(tokens)

                        logger.warn(
                            "  ⚠️ Segment has no word timestamps, estimating...",
                            tokens_count=len(tokens),
                            estimated_word_duration_ms=f"{token_duration * 1000:.1f}"
                        )

                        for idx, token in enumerate(tokens):
                            segment_words += 1
                            start = segment_start_time + idx * token_duration
                            end = start + token_duration
                            word_list.append(Word(
                                word=token,
                                start=start,
                                end=end,
                                confidence=segment_confidence
                            ))

                logger.info(
                    f"  ✅ Segment {segment_idx + 1} processed",
                    words_extracted=segment_words
                )

                full_text += segment_text + " "

            full_text = full_text.strip()
            segment_process_time = time.time() - segment_start

            logger.info(
                "✅ Segment processing complete",
                total_segments=total_segments,
                total_words=len(word_list),
                process_time_seconds=f"{segment_process_time:.3f}s"
            )

            # Calculate overall confidence
            confidence = info.language_probability if hasattr(info, 'language_probability') else 0.5
            if segment_list and any(s.confidence is not None for s in segment_list):
                confidences = [s.confidence for s in segment_list if s.confidence is not None]
                if confidences:
                    confidence = sum(confidences) / len(confidences)

            total_duration = time.time() - start_time
            mem_after = self._get_memory_info()

            # Calculate word count
            word_count = len(word_list)
            token_count = len(full_text.split())

            logger.info(
                "🎉 TRANSCRIPTION COMPLETE - SUMMARY",
                total_duration_seconds=f"{total_duration:.3f}s",
                breakdown={
                    "audio_loading": f"{audio_load_time:.3f}s",
                    "transcription": f"{transcribe_time:.3f}s",
                    "processing": f"{segment_process_time:.3f}s"
                },
                text_length=len(full_text),
                full_text=full_text,
                word_count=word_count,
                token_count=token_count,
                confidence=f"{confidence:.3f}",
                segments=len(segment_list),
                language=info.language if hasattr(info, 'language') else 'en',
                language_probability=f"{info.language_probability:.3f}" if hasattr(info, 'language_probability') else None,
                audio_duration_seconds=f"{info.duration:.3f}" if hasattr(info, 'duration') else None,
                memory_before_mb=f"{mem_before['rss_mb']:.1f}",
                memory_after_mb=f"{mem_after['rss_mb']:.1f}",
                memory_change_mb=f"{mem_after['rss_mb'] - mem_before['rss_mb']:.1f}"
            )

            return TranscriptionResponse(
                text=full_text,
                language=info.language if hasattr(info, 'language') else 'en',
                confidence=confidence,
                duration=total_duration,
                segments=segment_list,
                words=word_list
            )

        except MemoryError as e:
            mem_error = self._get_memory_info()
            logger.error(
                "❌ OUT OF MEMORY during transcription",
                error=str(e),
                memory_info=mem_error,
                audio_path=audio_path,
                duration_seconds=f"{time.time() - start_time:.3f}"
            )
            raise RuntimeError("Transcription service out of memory. Please try with shorter audio or restart service.")
        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "❌ TRANSCRIPTION FAILED",
                audio_path=audio_path,
                duration_seconds=f"{duration:.3f}",
                error=str(e)
            )
            raise