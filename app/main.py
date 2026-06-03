import structlog
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import config
from services.transcription_service import TranscriptionService
from services.acoustic_analysis_service import AcousticAnalysisService
from schemas.acoustic import AcousticAnalysisRequest, AcousticAnalysisResponse
from schemas.transcription import TranscriptionRequest, TranscriptionResponse, HealthResponse

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global service instance
transcription_service = None
acoustic_analysis_service = None
request_metrics = {
    "requests": 0,
    "errors": 0,
    "totalLatencyMs": 0,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_service, acoustic_analysis_service
    # Startup
    logger.info("Starting speech worker")
    transcription_service = TranscriptionService()
    acoustic_analysis_service = AcousticAnalysisService()
    yield
    # Shutdown
    logger.info("Shutting down speech worker")

app = FastAPI(
    title="Speech Transcription Worker",
    description="FastAPI service for speech-to-text transcription using faster-whisper",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root health endpoint for probes hitting the service base URL."""
    model_loaded = transcription_service is not None and transcription_service.model is not None
    return {
        "success": True,
        "status": "healthy" if model_loaded else "unhealthy",
        "message": "Speech worker is running",
        "healthUrl": "/health",
        "metricsUrl": "/metrics",
    }


@app.get("/healthz")
async def healthz():
    """Compatibility health endpoint for probes expecting a root-style path."""
    model_loaded = transcription_service is not None and transcription_service.model is not None
    return {
        "success": True,
        "status": "healthy" if model_loaded else "unhealthy",
        "message": "Speech worker is running",
    }


@app.middleware("http")
async def metrics_middleware(request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        request_metrics["requests"] += 1
        request_metrics["errors"] += 1
        request_metrics["totalLatencyMs"] += elapsed_ms
        raise

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    request_metrics["requests"] += 1
    request_metrics["totalLatencyMs"] += elapsed_ms
    if response.status_code >= 400:
        request_metrics["errors"] += 1
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = transcription_service is not None and transcription_service.model is not None
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        device=config.WHISPER_DEVICE
    )


@app.get("/metrics")
async def metrics():
    """Request telemetry for the worker"""
    return {
        "success": True,
        "data": {
            "speech-worker": {
                "requests": request_metrics["requests"],
                "errors": request_metrics["errors"],
                "totalLatencyMs": request_metrics["totalLatencyMs"],
            }
        },
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """Transcribe audio file"""
    try:
        if not transcription_service:
            raise HTTPException(status_code=503, detail="Transcription service not available")

        result = await transcription_service.transcribe(request.audioPath)
        return result

    except FileNotFoundError:
        logger.error("Audio file not found", audio_path=request.audioPath)
        raise HTTPException(status_code=404, detail="Audio file not found")
    except Exception as e:
        logger.error("Transcription failed", error=str(e), audio_path=request.audioPath)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/acoustic/phonemes", response_model=AcousticAnalysisResponse)
async def analyze_phoneme_acoustics(request: AcousticAnalysisRequest):
    """Extract acoustic features for externally aligned phoneme intervals."""
    try:
        if not acoustic_analysis_service:
            raise HTTPException(status_code=503, detail="Acoustic analysis service not available")

        return await acoustic_analysis_service.analyze(request)

    except FileNotFoundError:
        logger.error("Audio file not found for acoustic analysis", audio_path=request.audioPath)
        raise HTTPException(status_code=404, detail="Audio file not found")
    except Exception as e:
        logger.error("Acoustic analysis failed", error=str(e), audio_path=request.audioPath)
        raise HTTPException(status_code=500, detail=f"Acoustic analysis failed: {str(e)}")

