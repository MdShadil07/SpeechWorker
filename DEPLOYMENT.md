# Speech Worker Deployment Guide

This document provides comprehensive instructions for deploying the **Speech Transcription Worker** in production environments. The speech worker is a dedicated backend service built with Python and FastAPI, utilizing `faster-whisper` for efficient speech-to-text inference.

> [!IMPORTANT]
> The Speech Worker should be treated as a private, internal service. It must **never** be exposed directly to the public internet. All requests should route through your primary backend or API gateway.

---

## 1. System Requirements

### Hardware
* **CPU Deployment**: Minimum 4 CPU cores, 8GB RAM (performance depends heavily on the model size).
* **GPU Deployment (Recommended)**: NVIDIA GPU with at least 4GB VRAM (e.g., T4, A10g), CUDA toolkit installed.
* **Storage**: Sufficient space for S3/local mounts and downloaded Whisper models (~1-3GB depending on the model).

### Software
* **Docker & Docker Compose** (Recommended for all deployments)
* **Python 3.11** (If deploying bare-metal)
* **FFmpeg** (Required for audio processing)
* **NVIDIA Container Toolkit** (If using GPU with Docker)

---

## 2. Configuration (Environment Variables)

The worker is configured via environment variables. In a Docker setup, these can be passed in your `.env` file or directly in the `docker-compose.yml`.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `WHISPER_MODEL` | `small.en` | The Whisper model to load (e.g., `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v2`). |
| `WHISPER_DEVICE` | `cpu` | Hardware accelerator: `cpu` or `cuda`. |
| `WHISPER_COMPUTE_TYPE` | `int8` | Precision. Use `int8` for CPU or `float16` / `int8_float16` for GPU. |
| `HOST` | `0.0.0.0` | Bind address for the FastAPI server. |
| `PORT` | `8001` | Port for the FastAPI server. |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## 3. Deployment Methods

### Method 1: Docker Compose (Recommended)

Using Docker is the safest and most reproducible method. It encapsulates the complex Python dependencies and system packages like `ffmpeg`.

1. **Navigate to the Speech Worker Directory:**
   ```bash
   cd backend/speech-worker
   ```

2. **Configure your `docker-compose.yml` (Included by default):**
   Ensure the volume mount paths correctly point to where your main backend stores uploaded audio files.
   ```yaml
   version: '3.8'
   services:
     speech-worker:
       build: .
       environment:
         WHISPER_MODEL: small.en
         WHISPER_DEVICE: cpu
         WHISPER_COMPUTE_TYPE: int8
         LOG_LEVEL: INFO
       volumes:
         # CRITICAL: Adjust this path to match your primary backend's upload directory
         - ../../uploads:/app/uploads:ro 
       ports:
         - "8001:8001" # Only expose to internal network in production
       restart: unless-stopped
   ```

3. **Start the Service:**
   ```bash
   docker-compose up -d --build
   ```

### Method 2: Native / Bare Metal

If you cannot use Docker, deploy directly using Python.

1. **Install System Dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install -y ffmpeg git
   ```

2. **Setup Python Virtual Environment:**
   ```bash
   cd backend/speech-worker
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python Packages:**
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

4. **Run via Uvicorn (Production):**
   ```bash
   # Note: ONLY use 1 worker per instance to prevent loading the model into RAM multiple times.
   uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1
   ```

---

## 4. GPU Acceleration (CUDA)

For high-throughput environments, GPU acceleration is strongly recommended. 

> [!TIP]
> GPU inference is typically 5x to 10x faster than CPU inference for Whisper models.

**Docker Compose changes for GPU:**
```yaml
services:
  speech-worker:
    build: .
    environment:
      WHISPER_MODEL: small.en
      WHISPER_DEVICE: cuda          # Changed from cpu
      WHISPER_COMPUTE_TYPE: float16 # Recommended for GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## 5. Architectural Best Practices

### The "One Worker Rule"
The Docker container is strictly designed to run **one Uvicorn worker** per container (`--workers 1`). 
Whisper models are large and load directly into RAM/VRAM. Running multiple workers in a single container will duplicate the model in memory, quickly leading to Out-Of-Memory (OOM) crashes.

### Scaling out
To handle more concurrent transcription requests:
1. Do **not** increase the Uvicorn worker count.
2. Instead, spin up **multiple instances/containers** of the `speech-worker`.
3. Put them behind a load balancer (like NGINX, HAProxy, or a Kubernetes Service) operating in a Round-Robin configuration.

### File Access
The speech worker does not handle file uploads over the network. Your primary backend saves the file to a shared volume, S3, or temp directory, and sends the *file path* to the worker. 
* Make sure the `speech-worker` has **Read-Only (ro)** access to the shared file directory.

---

## 6. Health Checks & Monitoring

The service includes a built-in health check endpoint.

**Endpoint:** `GET http://localhost:8001/health`

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

**Docker Healthcheck integration:**
This is already included in the `docker-compose.yml`, but if running in Kubernetes or AWS ECS, configure the HTTP probe to check `/health` on port `8001` every 30 seconds.

---

## 7. Troubleshooting

* **Out of Memory (OOM) Killed:** 
  * Cause: The selected `WHISPER_MODEL` is too large for your RAM/VRAM, or you are running multiple Uvicorn workers.
  * Fix: Downgrade the model (e.g., `medium` -> `small`), or ensure `--workers 1` is set.
* **FFmpeg Not Found:**
  * Cause: The system lacks `ffmpeg`.
  * Fix: Ensure `apt-get install -y ffmpeg` ran successfully (included in Dockerfile).
* **CUDA Device Not Found:**
  * Cause: NVIDIA drivers or container toolkit is missing/misconfigured.
  * Fix: Verify `nvidia-smi` works on the host and that the docker daemon is configured with the `nvidia` runtime.
