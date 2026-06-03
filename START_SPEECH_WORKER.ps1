# Speech Worker Startup Script with Memory Optimization (PowerShell)

Write-Host ""
Write-Host "=========================================="
Write-Host "Speech Worker - Optimized Startup"
Write-Host "=========================================="
Write-Host ""

# Set memory optimization environment variables
Write-Host "Setting memory optimization environment variables..."
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:MKL_THREADING_LAYER = "GNU"
$env:NUMEXPR_MAX_THREADS = "1"

# Optional: Override Whisper model if needed
# $env:WHISPER_MODEL = "tiny.en"
# $env:WHISPER_COMPUTE_TYPE = "int8"

Write-Host ""
Write-Host "Environment Variables Set:"
Write-Host "  OMP_NUM_THREADS=$env:OMP_NUM_THREADS"
Write-Host "  MKL_NUM_THREADS=$env:MKL_NUM_THREADS"
Write-Host "  NUMEXPR_NUM_THREADS=$env:NUMEXPR_NUM_THREADS"
Write-Host "  MKL_THREADING_LAYER=$env:MKL_THREADING_LAYER"
Write-Host "  NUMEXPR_MAX_THREADS=$env:NUMEXPR_MAX_THREADS"
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..."
Push-Location "D:\ENGLISH PRACTICE\Learn English\backend"
& "..\..\..\venv\Scripts\Activate.ps1"

if (-not $?) {
    Write-Host "Error: Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Starting Speech Worker on port 8001..."
Write-Host ""

# Start the uvicorn server
cd speech-worker
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
