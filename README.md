# Speech Transcription Worker

FastAPI-based speech-to-text transcription service using faster-whisper for the English Learning Platform.

This container is intended to run as a private inference tier behind the API and orchestrator workers.

## Features

- Local faster-whisper integration
- FastAPI REST API
- Structured logging
- Health monitoring
- GPU-ready architecture
- Docker containerization

## Quick Start

### Using Docker Compose

```bash
cd speech-worker
docker-compose up --build
```

Production containers should be started with uvicorn directly from the Docker image. The container is designed to run one model-loaded process per container.

### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
python -m app.main
```

or, in production containers:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1
```

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### POST /transcribe
Transcribe audio file.

**Request:**
```json
{
  "audioPath": "/path/to/audio.wav"
}
```

**Response:**
```json
{
  "text": "Hello world",
  "language": "en",
  "confidence": 0.94,
  "duration": 1.23,
  "segments": [
    {
      "start": 0.0,
      "end": 1.0,
      "text": "Hello",
      "confidence": 0.95
    }
  ]
}
```

## Configuration

Environment variables:

- `WHISPER_MODEL`: Model size (default: small.en)
- `WHISPER_DEVICE`: Device (cpu/cuda, default: cpu)
- `WHISPER_COMPUTE_TYPE`: Compute type (default: int8 in production)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8001)
- `LOG_LEVEL`: Logging level (default: INFO)

Operational notes:

- run one model-heavy worker per container
- keep the container private behind the orchestrator tier
- mount audio artifacts from S3 or a temporary workspace only
- do not expose this service publicly
- if you use AWS S3 direct upload, the bucket must allow CORS for browser PUT requests from the frontend origin

## GPU Support

For GPU deployment, set:
- `WHISPER_DEVICE=cuda`
- `WHISPER_COMPUTE_TYPE=float16` (or int8 for faster inference)

Ensure CUDA is available in the container.

## Compatibility Notes

- The API sends the inference service the audio path or object key already stored during upload-session completion.
- The frontend upload flow remains unchanged for the current version: create session, upload chunks, complete upload, submit attempt, poll result.